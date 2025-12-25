
import os
import argparse
import requests
import json
from dotenv import load_dotenv

load_dotenv()


def parse_pdf_with_landingai(pdf_path: str, api_key: str, model: str = "dpt-2"):
    print(f"Parsing PDF with LandingAI (model: {model})...")
    print(f"Input: {pdf_path}")
    
    headers = {"Authorization": f"Basic {api_key}"}
    
    with open(pdf_path, "rb") as pdf_file:
        parse_response = requests.post(
            url="https://api.va.landing.ai/v1/ade/parse",
            headers=headers,
            files=[("document", pdf_file)],
            data={"model": model}
        )
    
    if parse_response.status_code != 200:
        raise Exception(f"LandingAI API error: {parse_response.status_code} - {parse_response.text}")

    return parse_response.json()


def save_json(data, output_path: str):
    print(f"Saving JSON to: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Parse PDF using LandingAI and save as JSON")
    parser.add_argument("--pdf-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model", type=str, default="dpt-2")
    
    args = parser.parse_args()
    
    landingai_api_key = os.getenv("LANDINGAI_API_KEY")
    if not landingai_api_key:
        raise ValueError("LANDINGAI_API_KEY must be set in .env file")
    
    try:
        parsed_data = parse_pdf_with_landingai(args.pdf_path, landingai_api_key, args.model)
        save_json(parsed_data, args.output_path)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

# python parse_landingai.py --pdf-path /Users/santhosh/Documents/study_projects/ringcentral_assessment/feature_extraction/pdfs_for_landing_ai/astor_manual_181_266.pdf --output-path /Users/santhosh/Documents/study_projects/ringcentral_assessment/feature_extraction/pdfs_for_landing_ai/astor_manual_181_266.json