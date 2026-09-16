import os
import json
import time
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import APIError

class FinancialDataLoader:
    def __init__(self, dataset_path: str = None, api_key: str = None):
        if dataset_path is None:
            self.base_path = Path(__file__).parent.parent / "dataset"
        else:
            self.base_path = Path(dataset_path)
        
        self.code_path = Path(__file__).parent
        self.cache_file = self.code_path / "extracted_image_amounts.json"
        
        # Load core files
        self.requests = pd.read_csv(self.base_path / "requests.csv")
        self.events = pd.read_csv(self.base_path / "financial_events.csv")
        self.profiles = pd.read_csv(self.base_path / "financial_profiles.csv")
        self.exchange_rates = pd.read_csv(self.base_path / "exchange_rates.csv")
        self.payment_options = pd.read_csv(self.base_path / "request_payment_options.csv")
        
        # Load context files
        self.messages = pd.read_csv(self.base_path / "messages.csv")
        self.images_df = pd.read_csv(self.base_path / "images.csv")
        
        # Load cached image extractions if they exist
        self.image_cache = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.image_cache = json.load(f)
            except Exception:
                self.image_cache = {}
        
        try:
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = genai.Client()
        except ValueError:
            print("Warning: No GEMINI_API_KEY provided. Image extraction will be skipped.")
            self.client = None

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.image_cache, f, indent=2)

    def get_user_profile(self, user_id: str) -> dict:
        profile = self.profiles[self.profiles['user_id'] == user_id]
        return profile.iloc[0].to_dict() if not profile.empty else {}
        
    def get_events_for_user(self, user_id: str) -> pd.DataFrame:
        return self.events[self.events['user_id'] == user_id]

    def get_request_details(self, request_id: str) -> dict:
        req = self.requests[self.requests['request_id'] == request_id]
        return req.iloc[0].to_dict() if not req.empty else {}

    def _extract_amount_from_image(self, event_id: str) -> float | None:
        if event_id in self.image_cache:
            return self.image_cache[event_id]

        if self.client is None:
            return None
            
        linked_image_row = self.images_df[self.images_df['related_event_id'] == event_id]
        if linked_image_row.empty:
            return None
            
        image_id = linked_image_row.iloc[0]['image_id']
        image_path = self.base_path / "media" / "images" / f"{image_id}.png"
        
        if not image_path.exists():
            print(f"Warning: Image file not found at {image_path}")
            return None

        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )
        
        prompt = (
            "Extract the total financial amount shown in this receipt or document. "
            "Respond ONLY with the numerical value (e.g., '145.50'). "
            "Do not include currency symbols, commas, or any other text."
        )

        # Retry loop for 429 rate limit
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=[prompt, image_part]
                )
                clean_text = response.text.strip().replace(',', '')
                amount = float(clean_text)
                
                # Cache result
                self.image_cache[event_id] = amount
                self._save_cache()
                
                # Small pause to stay within 5 requests/minute quota
                time.sleep(12)
                return amount
                
            except APIError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 40  # Default suggested by API response
                    print(f"Rate limit hit on {image_id}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"API error for {image_id}: {e}")
                    return None
            except ValueError:
                print(f"Warning: Could not parse float from Gemini response for {image_id}")
                return None
            except Exception as e:
                print(f"Warning: Unexpected failure on {image_id}: {e}")
                return None
                
        return None

    def resolve_missing_amounts(self):
        missing_mask = self.events['amount'].isna()
        missing_rows = self.events[missing_mask]
        print(f"Found {len(missing_rows)} events with missing amounts.")
        
        for index, row in missing_rows.iterrows():
            event_id = row['event_id']
            extracted_amount = self._extract_amount_from_image(event_id)
            
            if extracted_amount is not None:
                self.events.at[index, 'amount'] = extracted_amount
                print(f"Resolved missing amount for {event_id}: {extracted_amount}")

        self.events['amount'] = self.events['amount'].astype(float)