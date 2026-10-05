import pathlib
root = pathlib.Path(r'C:/Users/hp/Pictures/Microfinince workers agent')
for p in root.rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.py', '.md', '.txt', '.json', '.ts', '.tsx', '.js', '.jsx'}:
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        if 'Supported formats: .pdf, .xlsx, .xls, .docx, .tiff, .tif, .png, .jpg' in text:
            print(p)
