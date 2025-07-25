from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware  # Optional but helpful
from model import predict_text  # Your ML function

app = FastAPI()

# Optional: CORS middleware (for frontend JS to access backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:8000"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (like images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# If frontend uses fetch/JS to POST JSON
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    result = predict_text(text)
    return JSONResponse(content={"result": result})
