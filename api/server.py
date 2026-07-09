# server.py
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional, Literal
from dotenv import load_dotenv

# Explicitly load the .env file
load_dotenv() 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://lemonadenotebook.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Reference(BaseModel):
    text: str
    link: Optional[str] = None


class ClassifyRequest(BaseModel):
    references: List[Reference]


@app.post("/api/classify-sources")
def classify_sources(request: ClassifyRequest):
    try:
        response = client.responses.create(
            model="gpt-5.5",
            input=[
                {
                    "role": "system",
                    "content": "You classify Wikipedia references. Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": f"""
Classify these Wikipedia references.

For each source, return:
- title
- url
- sourceType: primary | secondary | tertiary | unclear
- format: book | journal | website | news | report | database | other
- paywallLikely: true | false | unclear
- reason

References:
{json.dumps([r.model_dump() for r in request.references], indent=2)}
""",
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "classified_sources",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": ["string", "null"]},
                                        "sourceType": {
                                            "type": "string",
                                            "enum": [
                                                "primary",
                                                "secondary",
                                                "tertiary",
                                                "unclear",
                                            ],
                                        },
                                        "format": {
                                            "type": "string",
                                            "enum": [
                                                "book",
                                                "journal",
                                                "website",
                                                "news",
                                                "report",
                                                "database",
                                                "other",
                                            ],
                                        },
                                        "paywallLikely": {
                                            "type": "string",
                                            "enum": ["true", "false", "unclear"],
                                        },
                                        "reason": {"type": "string"},
                                    },
                                    "required": [
                                        "title",
                                        "url",
                                        "sourceType",
                                        "format",
                                        "paywallLikely",
                                        "reason",
                                    ],
                                },
                            },
                        },
                        "required": ["sources"],
                    },
                }
            },
        )

        return json.loads(response.output_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))