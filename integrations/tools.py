import base64
import json
import mimetypes
import os
import smtplib
import subprocess
import sys
import tempfile
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class ToolError(Exception):
    pass


class ToolCredentialStore:
    FILE_PATH = Path("tool_credentials.json")

    @classmethod
    def load(cls) -> Dict[str, Any]:
        try:
            with cls.FILE_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
        return {}

    @classmethod
    def update(cls, values: Dict[str, Any]) -> None:
        payload = cls.load()
        payload.update({k: v for k, v in values.items() if v is not None})
        cls.FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with cls.FILE_PATH.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


class ToolBase:
    args_schema: Dict[str, Any] = {
        'type': 'object',
        'properties': {},
        'required': [],
    }

    credential_requirements: List[Dict[str, Any]] = []

    def runtime_config(self) -> Dict[str, Any]:
        config = dict(os.environ)
        config.update(ToolCredentialStore.load())
        return config

    def get_credential(self, key: str) -> Any:
        return self.runtime_config().get(key)

    def is_configured(self) -> bool:
        for requirement in self.credential_requirements:
            env = requirement.get('env')
            if isinstance(env, str):
                keys = [env]
            else:
                keys = env

            if requirement.get('any_of'):
                if any(self.get_credential(k) not in (None, '') for k in keys):
                    continue
                return False

            if requirement.get('required', True):
                if any(self.get_credential(k) in (None, '') for k in keys):
                    return False

        return True

    def missing_credentials(self) -> List[str]:
        missing: List[str] = []
        for requirement in self.credential_requirements:
            env = requirement.get('env')
            if isinstance(env, str):
                keys = [env]
            else:
                keys = env

            if requirement.get('any_of'):
                if all(self.get_credential(k) in (None, '') for k in keys):
                    missing.extend(keys)
            elif requirement.get('required', True):
                missing.extend([k for k in keys if self.get_credential(k) in (None, '')])

        return missing

    def credential_details(self) -> List[Dict[str, Any]]:
        return self.credential_requirements

    def __init__(self, name: str, description: str, cooldown: int = 15):
        self.name = name
        self.description = description
        self.cooldown = cooldown
        self.last_used_at: Optional[float] = None

    def __init__(self, name: str, description: str, cooldown: int = 15):
        self.name = name
        self.description = description
        self.cooldown = cooldown
        self.last_used_at: Optional[float] = None

    def can_run(self) -> bool:
        if self.last_used_at is None:
            return True
        return time.time() - self.last_used_at >= self.cooldown

    def update_last_used(self) -> None:
        self.last_used_at = time.time()

    @classmethod
    def to_openai_tool(cls) -> dict:
        instance = cls()
        return {
            'type': 'function',
            'function': {
                'name': instance.name,
                'description': instance.description,
                'parameters': instance.args_schema,
            },
        }

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class EmailSendTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['to', 'subject', 'body'],
        'properties': {
            'to': {'type': 'string', 'description': 'Recipient email address'},
            'subject': {'type': 'string', 'description': 'Email subject line'},
            'body': {'type': 'string', 'description': 'Email body text'},
        },
    }

    credential_requirements = [
        {'env': 'EMAIL_USERNAME', 'label': 'SMTP username'},
        {'env': 'EMAIL_PASSWORD', 'label': 'SMTP password', 'secret': True},
    ]

    def __init__(self):
        super().__init__(name='email_send', description='Send email via SMTP', cooldown=10)

    def execute(self, to: str, subject: str, body: str, **kwargs: Any) -> Dict[str, Any]:
        smtp_server = self.get_credential('EMAIL_SMTP_SERVER') or 'smtp.gmail.com'
        smtp_port = int(self.get_credential('EMAIL_SMTP_PORT') or '587')
        username = self.get_credential('EMAIL_USERNAME')
        password = self.get_credential('EMAIL_PASSWORD')
        if not username or not password:
            raise ToolError('EMAIL_USERNAME and EMAIL_PASSWORD must be set for SMTP access')
        message = EmailMessage()
        message['From'] = username
        message['To'] = to
        message['Subject'] = subject
        message.set_content(body)
        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(message)
        self.update_last_used()
        return {'status': 'sent', 'recipient': to, 'subject': subject}


class EmailInboxTool(ToolBase):
    args_schema = {
        'type': 'object',
        'properties': {
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 5},
        },
    }

    credential_requirements = [
        {'env': 'EMAIL_USERNAME', 'label': 'IMAP username'},
        {'env': 'EMAIL_PASSWORD', 'label': 'IMAP password', 'secret': True},
    ]

    def __init__(self):
        super().__init__(name='email_inbox', description='Read recent email headers using IMAP', cooldown=30)

    def execute(self, limit: int = 5, **kwargs: Any) -> Dict[str, Any]:
        try:
            import imaplib
        except ImportError:
            raise ToolError('imaplib is unavailable in this environment')
        imap_server = self.get_credential('EMAIL_IMAP_SERVER') or 'imap.gmail.com'
        username = self.get_credential('EMAIL_USERNAME')
        password = self.get_credential('EMAIL_PASSWORD')
        if not username or not password:
            raise ToolError('EMAIL_USERNAME and EMAIL_PASSWORD must be set for IMAP access')
        with imaplib.IMAP4_SSL(imap_server) as imap:
            imap.login(username, password)
            imap.select('INBOX')
            status, data = imap.search(None, 'ALL')
            if status != 'OK':
                raise ToolError('IMAP search failed')
            ids = data[0].split()
            ids = ids[-limit:]
            subjects: List[str] = []
            for msg_id in reversed(ids):
                status, msg_data = imap.fetch(msg_id, '(BODY[HEADER.FIELDS (SUBJECT)])')
                if status != 'OK':
                    continue
                parts = msg_data[0]
                if isinstance(parts, tuple):
                    subject_header = parts[1].decode('utf-8', errors='ignore')
                    subjects.append(subject_header.strip())
        self.update_last_used()
        return {'count': len(subjects), 'recent_subjects': subjects}


class PythonExecutionTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['code'],
        'properties': {
            'code': {'type': 'string', 'description': 'Python code to execute'},
            'timeout': {'type': 'integer', 'minimum': 1, 'maximum': 30, 'default': 30},
        },
    }

    def __init__(self):
        super().__init__(name='python_execute', description='Execute Python code in a safe isolated workspace', cooldown=10)

    def execute(self, code: str, work_dir: Optional[str] = None, timeout: int = 60, **kwargs: Any) -> Dict[str, Any]:
        workspace = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix='python_exec_'))
        workspace.mkdir(parents=True, exist_ok=True)
        script_path = workspace / 'script.py'
        script_path.write_text(code, encoding='utf-8')
        process = subprocess.run([
            sys.executable,
            str(script_path)
        ], cwd=str(workspace), capture_output=True, text=True, timeout=timeout)
        outputs = {
            'returncode': process.returncode,
            'stdout': process.stdout,
            'stderr': process.stderr,
            'workspace': str(workspace),
            'files': [str(p) for p in workspace.iterdir() if p.is_file()]
        }
        self.update_last_used()
        return outputs


class VisualizationTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['code'],
        'properties': {
            'code': {'type': 'string', 'description': 'Python code that generates a visualization using matplotlib'},
            'output_path': {'type': 'string', 'default': 'visualization.png'},
            'work_dir': {'type': 'string'},
            'timeout': {'type': 'integer', 'minimum': 1, 'default': 60},
        },
    }

    def __init__(self):
        super().__init__(name='visualization', description='Render visualization code and save image assets', cooldown=10)

    def execute(self, code: str, output_path: str = 'visualization.png', work_dir: Optional[str] = None, timeout: int = 60, **kwargs: Any) -> Dict[str, Any]:
        workspace = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix='visualization_'))
        workspace.mkdir(parents=True, exist_ok=True)
        script_path = workspace / 'visualization.py'
        wrapped_code = (
            'import matplotlib\n'
            'import matplotlib.pyplot as plt\n'
            'matplotlib.use("Agg")\n'
            + code + '\n'
        )
        script_path.write_text(wrapped_code, encoding='utf-8')
        process = subprocess.run([
            sys.executable,
            str(script_path)
        ], cwd=str(workspace), capture_output=True, text=True, timeout=timeout)
        output_file = workspace / output_path
        result = {
            'returncode': process.returncode,
            'stdout': process.stdout,
            'stderr': process.stderr,
            'output_file': str(output_file) if output_file.exists() else None,
            'workspace': str(workspace),
        }
        self.update_last_used()
        return result


class WordReportTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['title', 'sections'],
        'properties': {
            'title': {'type': 'string'},
            'sections': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'required': ['heading', 'content'],
                    'properties': {
                        'heading': {'type': 'string'},
                        'content': {'type': 'string'},
                    },
                },
            },
        },
    }

    def __init__(self):
        super().__init__(name='word_report', description='Create a Word report document with text and optional images', cooldown=20)

    def execute(self, title: str, sections: List[Dict[str, str]], images: Optional[List[str]] = None, output_path: str = 'report.docx', **kwargs: Any) -> Dict[str, Any]:
        doc = Document()
        doc.add_heading(title, level=0)
        for section in sections:
            heading = section.get('heading')
            content = section.get('content', '')
            if heading:
                doc.add_heading(heading, level=1)
            for paragraph in content.split('\n'):
                doc.add_paragraph(paragraph.strip())
        if images:
            for image_path in images:
                image_file = Path(image_path)
                if image_file.exists():
                    try:
                        doc.add_picture(str(image_file))
                    except Exception:
                        continue
        doc.save(output_path)
        self.update_last_used()
        return {'output_path': str(output_path), 'sections': len(sections), 'images': images or []}


class WebhookTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['message'],
        'properties': {
            'message': {'type': 'string'},
            'method': {'type': 'string', 'enum': ['POST', 'PUT'], 'default': 'POST'},
        },
    }

    def __init__(self, name: str, description: str, url_env: str, cooldown: int = 30):
        super().__init__(name=name, description=description, cooldown=cooldown)
        self.url_env = url_env
        self.credential_requirements = [
            {
                'env': url_env,
                'label': f'{name} webhook URL',
                'description': f'Webhook URL for {name}',
                'required': True,
            }
        ]

    def execute(self, message: str, method: str = 'POST', **kwargs: Any) -> Dict[str, Any]:
        webhook_url = self.get_credential(self.url_env)
        if not webhook_url:
            raise ToolError(f'{self.name} webhook is not configured. Set {self.url_env}.')
        import urllib.request
        import json
        payload = json.dumps({'text': message}).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=payload, headers={'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(req) as response:
            body = response.read().decode('utf-8')
        self.update_last_used()
        return {'status': 'posted', 'method': method, 'response': body}


class SlackTool(WebhookTool):
    args_schema = {
        'type': 'object',
        'required': ['message'],
        'properties': {
            'message': {'type': 'string'},
            'channel': {'type': 'string'},
        },
    }

    def __init__(self):
        super().__init__(name='slack_post', description='Post a message to Slack via webhook', url_env='SLACK_WEBHOOK_URL', cooldown=15)


class TeamsTool(WebhookTool):
    args_schema = {
        'type': 'object',
        'required': ['message'],
        'properties': {
            'message': {'type': 'string'},
            'channel': {'type': 'string'},
        },
    }

    def __init__(self):
        super().__init__(name='teams_post', description='Post a message to Microsoft Teams via webhook', url_env='TEAMS_WEBHOOK_URL', cooldown=15)


class GPToolMixin:
    SCOPES: List[str] = []

    credential_requirements = [
        {
            'env': ['GOOGLE_APPLICATION_CREDENTIALS', 'GOOGLE_SERVICE_ACCOUNT_JSON'],
            'label': 'Google service account credentials',
            'description': 'Either a path to a JSON key file or the raw JSON contents',
            'any_of': True,
            'required': True,
        },
        {
            'env': 'GOOGLE_IMPERSONATED_USER',
            'label': 'Google impersonated user',
            'description': 'Optional service account impersonated user',
            'required': False,
        },
    ]

    def get_credentials(self) -> Any:
        credentials_path = self.get_credential('GOOGLE_APPLICATION_CREDENTIALS')
        service_account_json = self.get_credential('GOOGLE_SERVICE_ACCOUNT_JSON')

        if not credentials_path and service_account_json:
            try:
                credentials_data = json.loads(service_account_json)
            except json.JSONDecodeError:
                try:
                    credentials_data = json.loads(
                        base64.b64decode(service_account_json).decode('utf-8')
                    )
                except Exception:
                    raise ToolError('GOOGLE_SERVICE_ACCOUNT_JSON must be valid JSON or base64-encoded JSON')
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False, encoding='utf-8'
            )
            json.dump(credentials_data, temp_file)
            temp_file.close()
            credentials_path = temp_file.name

        if not credentials_path or not Path(credentials_path).exists():
            raise ToolError('GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_SERVICE_ACCOUNT_JSON must be configured for Google API access')

        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=self.SCOPES
        )
        subject = self.get_credential('GOOGLE_IMPERSONATED_USER')
        if subject:
            creds = creds.with_subject(subject)
        if not creds.valid:
            creds.refresh(Request())
        return creds


class GoogleDriveTool(ToolBase, GPToolMixin):
    args_schema = {
        'type': 'object',
        'required': ['local_path'],
        'properties': {
            'local_path': {'type': 'string', 'description': 'Local file path to upload to Google Drive'},
            'mime_type': {'type': 'string'},
            'folder_id': {'type': 'string'},
        },
    }
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    def __init__(self):
        super().__init__(name='google_drive_upload', description='Upload a file to Google Drive', cooldown=30)

    def execute(self, local_path: str, mime_type: Optional[str] = None, folder_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        local_path_obj = Path(local_path)
        if not local_path_obj.exists():
            raise ToolError(f'Local file does not exist: {local_path}')
        creds = self.get_credentials()
        drive = build('drive', 'v3', credentials=creds, cache_discovery=False)
        file_metadata = {'name': local_path_obj.name}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        mime_type = mime_type or mimetypes.guess_type(local_path)[0] or 'application/octet-stream'
        media = MediaFileUpload(str(local_path_obj), mimetype=mime_type)
        file = drive.files().create(body=file_metadata, media_body=media, fields='id, name, mimeType, parents').execute()
        self.update_last_used()
        return {'file_id': file.get('id'), 'name': file.get('name'), 'mime_type': file.get('mimeType')}


class GoogleSheetsTool(ToolBase, GPToolMixin):
    args_schema = {
        'type': 'object',
        'required': ['spreadsheet_id', 'values'],
        'properties': {
            'spreadsheet_id': {'type': 'string'},
            'sheet_range': {'type': 'string', 'default': 'Sheet1!A:Z'},
            'values': {'type': 'array', 'items': {'type': 'array'}},
            'value_input_option': {'type': 'string', 'enum': ['RAW', 'USER_ENTERED'], 'default': 'RAW'},
        },
    }
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    def __init__(self):
        super().__init__(name='google_sheets_append', description='Append rows to a Google Sheet', cooldown=20)

    def execute(self, spreadsheet_id: str, sheet_range: str, values: List[List[Any]], value_input_option: str = 'RAW', **kwargs: Any) -> Dict[str, Any]:
        if not values:
            raise ToolError('Values cannot be empty')
        creds = self.get_credentials()
        sheets = build('sheets', 'v4', credentials=creds, cache_discovery=False)
        body = {'values': values}
        result = sheets.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=sheet_range,
            valueInputOption=value_input_option,
            insertDataOption='INSERT_ROWS',
            body=body,
        ).execute()
        self.update_last_used()
        return {'updates': result.get('updates', {})}


class GoogleCalendarTool(ToolBase, GPToolMixin):
    args_schema = {
        'type': 'object',
        'required': ['summary'],
        'properties': {
            'summary': {'type': 'string'},
            'start_datetime': {'type': 'string', 'format': 'date-time'},
            'end_datetime': {'type': 'string', 'format': 'date-time'},
            'start_time': {'type': 'string', 'format': 'date-time'},
            'end_time': {'type': 'string', 'format': 'date-time'},
            'attendees': {'type': 'array', 'items': {'type': 'string'}},
            'description': {'type': 'string'},
            'calendar_id': {'type': 'string', 'default': 'primary'},
            'timezone': {'type': 'string', 'default': 'UTC'},
        },
        'anyOf': [
            {'required': ['start_datetime', 'end_datetime']},
            {'required': ['start_time', 'end_time']},
        ],
    }
    SCOPES = ['https://www.googleapis.com/auth/calendar']

    def __init__(self):
        super().__init__(name='google_calendar_event', description='Create a Google Calendar event', cooldown=20)

    def execute(self, summary: str, start_datetime: Optional[str] = None, end_datetime: Optional[str] = None, timezone: str = 'UTC', attendees: Optional[List[str]] = None, description: Optional[str] = None, calendar_id: str = 'primary', **kwargs: Any) -> Dict[str, Any]:
        if not start_datetime:
            start_datetime = kwargs.get('start_time')
        if not end_datetime:
            end_datetime = kwargs.get('end_time')
        if not start_datetime or not end_datetime:
            raise ToolError('start_datetime/end_datetime or start_time/end_time are required')
        creds = self.get_credentials()
        service = build('calendar', 'v3', credentials=creds, cache_discovery=False)
        event = {
            'summary': summary,
            'description': description or '',
            'start': {'dateTime': start_datetime, 'timeZone': timezone},
            'end': {'dateTime': end_datetime, 'timeZone': timezone},
        }
        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]
        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        self.update_last_used()
        return {'event_id': created.get('id'), 'summary': created.get('summary')}


class PaymentReminderTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['customer_name', 'loan_id', 'amount_due', 'due_date'],
        'properties': {
            'customer_name': {'type': 'string'},
            'to': {'type': 'string'},
            'send_to': {'type': 'string'},
            'loan_id': {'type': 'string'},
            'amount_due': {'type': 'string'},
            'due_date': {'type': 'string', 'format': 'date'},
            'additional_details': {'type': 'string'},
        },
        'anyOf': [
            {'required': ['to']},
            {'required': ['send_to']},
        ],
    }

    def __init__(self):
        super().__init__(name='payment_reminder', description='Send loan payment reminder emails to clients', cooldown=45)
        self.email_tool = EmailSendTool()

    def execute(self, customer_name: str, loan_id: str, amount_due: str, due_date: str, to: Optional[str] = None, send_to: Optional[str] = None, customer_id: Optional[str] = None, additional_details: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        recipient = to or send_to
        if not recipient:
            raise ToolError('Recipient email address is required via to or send_to')
        subject = f'Payment Reminder for Loan #{loan_id}'
        body = (
            f'Dear {customer_name},\n\n'
            f'This is a reminder that your loan payment of {amount_due} is due on {due_date}.\n\n'
            f'Loan ID: {loan_id}\n'
            f'{additional_details or "Please settle the payment on time to avoid penalties."}\n\n'
            'If you need assistance, please contact our branch support team.\n\n'
            'Thank you,\n'
            'Microfinance Customer Care'
        )
        self.email_tool.execute(to=recipient, subject=subject, body=body)
        self.update_last_used()
        return {'status': 'reminder_sent', 'recipient': recipient, 'loan_id': loan_id, 'customer_id': customer_id}


class ClientOnboardingTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['customer_id', 'sheet_id', 'sheet_range'],
        'properties': {
            'customer_name': {'type': 'string'},
            'client_name': {'type': 'string'},
            'customer_id': {'type': 'string'},
            'product_type': {'type': 'string'},
            'loan_product': {'type': 'string'},
            'branch_name': {'type': 'string'},
            'sheet_id': {'type': 'string'},
            'sheet_range': {'type': 'string'},
            'send_welcome': {'type': 'boolean', 'default': False},
            'welcome_message': {'type': 'string'},
            'email': {'type': 'string'},
        },
        'anyOf': [
            {'required': ['customer_name', 'product_type']},
            {'required': ['client_name', 'product_type']},
            {'required': ['customer_name', 'loan_product']},
            {'required': ['client_name', 'loan_product']},
        ],
    }

    def __init__(self):
        super().__init__(name='client_onboarding', description='Record new client details in Google Sheets and optionally send a welcome email', cooldown=60)
        self.sheet_tool = GoogleSheetsTool()
        self.email_tool = EmailSendTool()

    def execute(self, customer_name: Optional[str], customer_id: str, product_type: Optional[str], branch_name: str, sheet_id: str, sheet_range: str, send_welcome: bool = False, welcome_message: Optional[str] = None, email: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        customer_name = customer_name or kwargs.get('client_name') or 'New Client'
        product_type = product_type or kwargs.get('loan_product') or 'Unknown Product'
        row = [[customer_id, customer_name, product_type, branch_name, str(time.time())]]
        sheet_result = self.sheet_tool.execute(spreadsheet_id=sheet_id, sheet_range=sheet_range, values=row)
        email_result = None
        if send_welcome:
            if not email:
                raise ToolError('email is required when send_welcome is True')
            subject = f'Welcome to {branch_name} Microfinance'
            body = welcome_message or (
                f'Dear {customer_name},\n\n'
                'Welcome to our microfinance services. We are happy to support your financial journey.\n\n'
                'Please contact us if you have questions.\n\n'
                f'Regards,\n{branch_name} Support Team'
            )
            email_result = self.email_tool.execute(to=email, subject=subject, body=body)
        self.update_last_used()
        result: Dict[str, Any] = {'sheet_write': sheet_result}
        if email_result:
            result['welcome_email'] = email_result
        return result


class RepaymentScheduleTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['customer_name', 'loan_id'],
        'properties': {
            'customer_name': {'type': 'string'},
            'loan_id': {'type': 'string'},
            'start_datetime': {'type': 'string', 'format': 'date-time'},
            'end_datetime': {'type': 'string', 'format': 'date-time'},
            'start_time': {'type': 'string', 'format': 'date-time'},
            'end_time': {'type': 'string', 'format': 'date-time'},
            'timezone': {'type': 'string', 'default': 'UTC'},
            'attendees': {'type': 'array', 'items': {'type': 'string'}},
            'description': {'type': 'string'},
        },
        'anyOf': [
            {'required': ['start_datetime', 'end_datetime']},
            {'required': ['start_time', 'end_time']},
        ],
    }

    def __init__(self):
        super().__init__(name='repayment_schedule', description='Schedule a repayment reminder event in Google Calendar', cooldown=30)
        self.calendar_tool = GoogleCalendarTool()

    def execute(self, customer_name: str, loan_id: str, start_datetime: Optional[str] = None, end_datetime: Optional[str] = None, timezone: str = 'UTC', attendees: Optional[List[str]] = None, description: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        if not start_datetime:
            start_datetime = kwargs.get('start_time')
        if not end_datetime:
            end_datetime = kwargs.get('end_time')
        if not start_datetime or not end_datetime:
            raise ToolError('start_datetime/end_datetime or start_time/end_time are required')
        summary = f'Repayment for {customer_name} (Loan {loan_id})'
        event_description = description or f'Repayment scheduled for loan {loan_id}. Please attend or pay on time.'
        event_result = self.calendar_tool.execute(
            summary=summary,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timezone=timezone,
            attendees=attendees,
            description=event_description,
        )
        self.update_last_used()
        return event_result


class LoanReportDriveTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['local_path'],
        'properties': {
            'local_path': {'type': 'string', 'description': 'Path to the report file to upload'},
            'folder_id': {'type': 'string'},
            'mime_type': {'type': 'string'},
        },
    }

    def __init__(self):
        super().__init__(name='loan_report_drive', description='Upload loan report files to Google Drive', cooldown=45)
        self.drive_tool = GoogleDriveTool()

    def execute(self, local_path: str, folder_id: Optional[str] = None, mime_type: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        result = self.drive_tool.execute(local_path=local_path, folder_id=folder_id, mime_type=mime_type)
        self.update_last_used()
        return result
