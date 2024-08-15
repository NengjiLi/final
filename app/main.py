from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.middleware.processPrompt import promptResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

class TrainingSummary(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, prompt: str = Form(...)):
    try:
        combined_prompt = promptResponse(prompt)
        input_ids = tokenizer(combined_prompt, return_tensors="pt").input_ids

        generated_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.7,
            max_length=100,
        )
        generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        final_text = generated_text.replace(combined_prompt, "").strip()

        words = final_text.split()
        final_text = ' '.join(words[:50])

        return templates.TemplateResponse("form.html", {"request": request, "response": final_text})

    except Exception as e:
        print(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Text generation failed")
