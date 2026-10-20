from datetime import datetime
import io
import json
import os
from pathlib import Path
import time
import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import errors, types
from PIL import Image
from pydantic import BaseModel, Field

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
  raise ValueError("GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=api_key)

app = FastAPI(title="Controlled QR/Barcode Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path(__file__).resolve().parent.parent / "products.db"


# Initialize Database Table on Startup
@app.on_event("startup")
async def init_db():
  async with aiosqlite.connect(DB_PATH) as db:
    await db.execute("""
            CREATE TABLE IF NOT EXISTS scanned_products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                barcode_number TEXT,
                pack_size TEXT,
                case_weight TEXT,
                best_before_end TEXT,
                price_mark TEXT,
                batch_or_ref_number TEXT,
                line_time_codes TEXT,
                manufacturer TEXT,
                scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    await db.commit()


# Pydantic Schema for Gemini
class ProductLabelData(BaseModel):
  product_name: str = Field(description="Full product name and flavor")
  pack_size: str = Field(description="Pack size and unit count, e.g. 500MLX12")
  price_mark: str = Field(
      description="Price mark if present, e.g. P1.29 or £1.29"
  )
  barcode_number: str = Field(
      description="Numeric digits printed beneath or encoded in the barcode"
  )
  batch_or_ref_number: str = Field(
      description="Batch or identification number, e.g. 250969"
  )
  best_before_end: str = Field(description="BBE date string exactly as printed")
  case_weight: str = Field(
      description="Total case weight including units (e.g. 6.39 kg)"
  )
  line_time_codes: str = Field(
      description="Production time and line codes like R72 11:51 6241"
  )
  manufacturer: str = Field(
      description="Manufacturer name and location printed at the bottom"
  )


static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_index():
  return FileResponse(static_dir / "index.html")


@app.post("/scan")
async def scan_and_save(image: UploadFile):
  try:
    image_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(image_bytes))

    prompt = """
        Examine this warehouse/product label carefully. 
        Extract all printed text and barcode digits into the specified structured schema.
        """

    candidate_models = [
        "gemini-2.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ]

    extracted_dict = None
    last_error = None

    for model_name in candidate_models:
      try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, pil_img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProductLabelData,
                temperature=0.1,
            ),
        )
        extracted_dict = json.loads(response.text)
        break
      except errors.ServerError as e:
        last_error = str(e)
        time.sleep(1)
        continue
      except errors.ClientError as e:
        last_error = str(e)
        continue

    if not extracted_dict:
      raise HTTPException(
          status_code=503, detail=f"All models unavailable: {last_error}"
      )

    # Save to SQLite Database
    async with aiosqlite.connect(DB_PATH) as db:
      cursor = await db.execute(
          """
                INSERT INTO scanned_products (
                    product_name, barcode_number, pack_size, case_weight,
                    best_before_end, price_mark, batch_or_ref_number,
                    line_time_codes, manufacturer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
          (
              extracted_dict.get("product_name"),
              extracted_dict.get("barcode_number"),
              extracted_dict.get("pack_size"),
              extracted_dict.get("case_weight"),
              extracted_dict.get("best_before_end"),
              extracted_dict.get("price_mark"),
              extracted_dict.get("batch_or_ref_number"),
              extracted_dict.get("line_time_codes"),
              extracted_dict.get("manufacturer"),
          ),
      )
      await db.commit()
      record_id = cursor.lastrowid

    extracted_dict["db_id"] = record_id
    return extracted_dict

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/records")
async def get_all_records():
  """Returns the list of all saved records from the database."""
  async with aiosqlite.connect(DB_PATH) as db:
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT * FROM scanned_products ORDER BY id DESC"
    ) as cursor:
      rows = await cursor.fetchall()
      return [dict(r) for r in rows]