# Resume Enhancer API

Layered FastAPI starter for a resume-enhancement service.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:APP_ENV = "development"
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

## Configuration

Set `APP_ENV` to `development`, `test`, or `production`. The application loads
`config/<environment>.yml`, then overlays matching environment variables. Put
local overrides and secrets in the repository's `.env` file; it is ignored by
Git and should never be committed. At minimum, set `OPENAI_API_KEY` before
calling the OpenAI-backed service.

The health endpoint is available at `GET /api/v1/health`.
