import re
from pathlib import Path
backend = Path(r'C:/Users/hp/Pictures/Microfinince workers agent/backend_agent_registry.py').read_text(encoding='utf-8')
frontend = Path(r'C:/Users/hp/Pictures/Microfinince  frontend/lib/agents.ts').read_text(encoding='utf-8')
backend_ids = re.findall(r'"([a-z0-9\-]+)": \{', backend)
frontend_ids = re.findall(r'id: "([a-z0-9\-]+)"', frontend)
backend_set = set(backend_ids)
frontend_set = set(frontend_ids)
print('frontend-only', sorted(frontend_set - backend_set))
print('backend-only', sorted(backend_set - frontend_set))
print('common count', len(frontend_set & backend_set))
print('frontend count', len(frontend_set), 'backend count', len(backend_set))
