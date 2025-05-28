from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Point to the 'templates' directory
templates = Jinja2Templates(directory="templates")

@app.get("/visualize", response_class=HTMLResponse)
async def visualize(request: Request):
    return templates.TemplateResponse("visualize.html", {"request": request})