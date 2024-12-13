import polars as pl
from datasets import load_dataset
import json
import requests
import os
import openai  
from json.decoder import JSONDecodeError
from tqdm import tqdm
import re

# GitHub API Key
GITHUB_TOKEN = "YOUR_TOKEN"
OPENAI_API_KEY = "YOUR_TOKEN"  
openai.api_key = OPENAI_API_KEY

def fetch_and_process_datasets():
    """
    Retrieves datasets, converts them to JSON format,
    and stores them in Polars DataFrames.
    """
    # Load datasets
    rust_code_explanation = load_dataset("Ak1104/Rust_Code_Explanation")
    rust_qa_final = load_dataset("Ak1104/Rust_QA_Final")

    # Convert to JSON
    rust_code_explanation_json = [
        {
            "output": record["output"],
            "input": record["input"],
            "instruction": record["instruction"]
        } for record in rust_code_explanation["train"]
    ]
    rust_qa_final_json = [
        {
            "Explanation": record["Explanation"],
            "Chapter": record["Chapter"],
            "Question": record["Question"]
        } for record in rust_qa_final["train"]
    ]

    # Create Polar DataFrames with a json_data column
    df_code_explanation = pl.DataFrame({"json_data": rust_code_explanation_json})
    df_qa_final = pl.DataFrame({"json_data": rust_qa_final_json})

    return df_code_explanation, df_qa_final

def generate_augmented_data_with_openai(target_rows=100):
    """
    Generates various augmented data using the OpenAI API.
    Appends data to the Polars DataFrame until the target number of rows is reached.

    Args:
    target_rows (int): Target number of rows in the DataFrame.

    Returns:
    pl.DataFrame: Polars DataFrame containing the augmented data.
    """

    # Prompt 
    prompt_template = """
    You are an expert in Rust programming and data augmentation.

    Your ONLY task is to return a strictly valid JSON array (no extra text, no code fences, no markdown) containing high-quality, unique, and realistic Rust code snippets.
    Return ONLY the JSON array with at least 3 objects, nothing else.

    IMPORTANT: 
    - The "input" field must be a single-line valid JSON string. This means:
    - Every newline in the Rust code must be replaced by the two-character sequence "\\n".
    - Every backslash character in the code must be escaped as `\\`.
    - Do not include any characters that would invalidate the JSON (e.g., `#\[` should become `#\\[` or `\\[` if needed).
    - Essentially, the Rust code should appear as a single line string with all internal newlines represented as "\\n" and all backslashes escaped as `\\`.
    - No actual newline characters are allowed inside the "input" field.
    - No invalid escape sequences are allowed.
    - No extra text outside the JSON array.

    Each JSON object in the array should have exactly the following string fields: 
    - "output": A clear, concise description (1-2 sentences) of what the code does and why it is useful.
    - "input": A single-line, fully compilable Rust code snippet demonstrating one or more Rust concepts (see list below). Must be properly escaped so it forms valid JSON.
    - "instruction": Brief instructions or context (1-2 sentences) on how or why to use or understand the code.

    Rust concepts to cover (at least one per snippet, ensure variety):
    1. Loops (for, while, loop)
    2. Structs and impl (with methods)
    3. Enums and match
    4. Traits and their implementation
    5. Macros (custom macros)
    6. Error Handling (Result, Option, unwrap, expect)
    7. Ownership and Borrowing
    8. Async/Await
    9. Module Systems (mod, use)
    10. Iterators
    11. Closures
    12. Generics

    Requirements:
    - Return only a strictly valid JSON array of at least 3 distinct objects. 
    - No explanation outside the JSON, no code fences, no markdown.
    - Each snippet must be unique, cover different combinations of Rust concepts, and be compilable.
    - All fields must be non-null strings.
    - The "input" field must contain a single-line valid JSON string. Replace every newline in the code with "\\n" and ensure every backslash is escaped as `\\`.
    - Code must follow Rust best practices and be realistic.
    - No duplicate code or identical snippets.

    Example (DO NOT COPY EXACTLY, but follow format):
    [
    {
        "output": "Implements a struct and a method to calculate the area of a rectangle.",
        "input": "struct Rectangle { width: u32, height: u32 } impl Rectangle { fn area(&self) -> u32 { self.width * self.height } } fn main() { let rect = Rectangle { width: 10, height: 5 }; println!(\"Area: {}\", rect.area()); }",
        "instruction": "Define a struct and implement a method to calculate its area, then create an instance and print the result."
    }
    ]

    Note in the actual answer:
    - The code snippet should have any internal newlines replaced by \"\\n\".
    - All backslashes must be escaped (e.g., `\\n`, `\\[`, `\\(`, etc.).
    - No invalid escape sequences.
    - Return only the JSON array.
    """
    augmented_data = []
    seen_inputs = set()

    while len(augmented_data) < target_rows:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Ajuster le modèle si nécessaire
                messages=[
                    {"role": "system", "content": "Return only a strictly valid JSON array."},
                    {"role": "user", "content": prompt_template}
                ],
                max_tokens=1000,
                temperature=1.0
            )
            
            raw_text = response.choices[0].message.content.strip()

            # Extracting the JSON array
            start_index = raw_text.find('[')
            end_index = raw_text.rfind(']')
            if start_index == -1 or end_index == -1:
                print(f"Unexpected response, no JSON array detected: {raw_text}")
                continue
            json_candidate = raw_text[start_index:end_index+1].strip()

            # Attempt to parse the JSON
            try:
                data_list = json.loads(json_candidate)
                if isinstance(data_list, list):
                    for data in data_list:
                        # Checking to avoid duplicates
                        if "input" in data and data["input"] not in seen_inputs:
                            # Checking for non-null fields
                            if all(field in data and data[field] is not None and isinstance(data[field], str) for field in ["output", "input", "instruction"]):
                                augmented_data.append(data)
                                seen_inputs.add(data["input"])
                                print(f"Number of data generated: {len(augmented_data)} / {target_rows}")
                                if len(augmented_data) >= target_rows:
                                    break
                            else:
                                print("Invalid JSON object or missing/null fields, ignored.")
                        else:
                            print("Duplicate detected, object ignored.")
                else:
                    print(f"Unexpected response (not listed):{json_candidate}")

            except json.JSONDecodeError as e:
                print(f"Error while parsing JSON: {e}, raw answer: {raw_text}")

        except Exception as e:
            print(f"Error generating data: {e}")

    # Truncate in case we have more than target_rows
    augmented_data = augmented_data[:target_rows]

    df_augmentation = pl.DataFrame({"json_data": augmented_data})
    return df_augmentation



def scrape_github_rust_files(github_org, max_files=300):
    """
    Scrapes .rs files from an organization's (or user's) GitHub repositories,
    by recursively traversing the directory tree.
    Attempts to retrieve strict JSON from the OpenAI API, without any fence or markdown code.

    Args:
    github_org (str): Name of the GitHub organization or user (e.g. "rust-lang")
    max_files (int): Maximum number of Rust files to scrape.

    Returns:
    pl.DataFrame: DataFrame containing the enriched JSON data.
    """
    file_count = 0
    rust_files_json = []

    progress_bar = tqdm(total=max_files, desc="Fetching Rust files", unit="file")

    api_url = f"https://api.github.com/orgs/{github_org}/repos"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    repos = response.json()

    for repo in repos:
        if file_count >= max_files:
            break

        repo_name = repo["full_name"]
        dirs_to_visit = [""]  # Commence par le répertoire racine

        while dirs_to_visit and file_count < max_files:
            current_path = dirs_to_visit.pop()
            url = f"https://api.github.com/repos/{repo_name}/contents/{current_path}" if current_path else f"https://api.github.com/repos/{repo_name}/contents"

            try:
                repo_response = requests.get(url, headers=headers)
                repo_response.raise_for_status()
                items = repo_response.json()

                for item in items:
                    if file_count >= max_files:
                        break

                    if item["type"] == "dir":
                        dirs_to_visit.append(item["path"])
                    else:
                        if item["name"].endswith(".rs"):
                            file_url = item["download_url"]
                            file_response = requests.get(file_url, headers=headers)
                            file_response.raise_for_status()

                            rust_code = file_response.text

                            # Clean up Rust code to remove control characters
                            cleaned_rust_code = re.sub(r'[\x00-\x1f\x7f]', '', rust_code)

                            # Prompt even stricter, formally prohibiting code fences
                            prompt = f"""
                            You are an expert in Rust programming and a strict JSON validator.

                            Your ONLY task is to produce a strictly valid JSON array describing the given Rust code.  
                            Follow these instructions very carefully:

                            1. **No code fences**, **no markdown formatting**, **no additional explanations** outside the JSON array.
                            2. **Return ONLY a valid JSON array**, nothing else. If you cannot produce a strictly valid JSON array according to the format below, return "[]".
                            3. Each JSON object in the array must contain exactly these fields:
                            - "output": A clear and detailed description of what the Rust code does. Use beginner-friendly language and highlight the main functionality.
                            - "input": The Rust code as a properly escaped JSON string. The code should be compilable, idiomatic, and formatted according to Rust standards (as if `rustfmt` was applied). If the original code is incomplete or non-compilable, fix it minimally to make it a complete, compilable snippet (e.g., add a `main` function if missing).
                            - "instruction": A concise explanation or context that helps a beginner understand how or why the code works, and any important Rust concepts demonstrated.

                            4. The returned JSON array must not contain any null fields. If you cannot guarantee this, return "[]".
                            5. If the snippet is very large or complex, produce a shorter, functional Rust snippet that captures the main idea, ensuring it still compiles.
                            6. No extra comments, no markdown, no code fences, no explanations outside the JSON array.

                            Example of a valid response:
                            [{{"output":"Prints 'Hello, world!' to the console","input":"fn main() {{ println!(\"Hello, world!\"); }}","instruction":"Basic Rust program that prints a string."}}]

                            Remember:
                            - Return ONLY a JSON array as per the instructions above.
                            - If you fail, return "[]".

                            Process this Rust code and return ONLY a JSON array:
                            {cleaned_rust_code}
                            """.strip()

                            try:
                                openai_response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a strict JSON validator that returns only valid JSON arrays."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=1500,
                                    temperature=0.7,
                                )

                                response_text = openai_response.choices[0].message.content.strip()

                                # If the response still contains fences, we remove them
                                # We will try to find the first '[' and the last ']'
                                # to isolate a valid JSON array.
                                if "```" in response_text:
                                    # Remove all code fences
                                    response_text = re.sub(r'```+.*?```+', '', response_text, flags=re.DOTALL).strip()

                                # Try to find a valid JSON portion
                                start_index = response_text.find('[')
                                end_index = response_text.rfind(']')
                                if start_index == -1 or end_index == -1:
                                    # No valid JSON array
                                    print("OpenAI response not compliant, no [ or ]:", response_text)
                                    raise ValueError("No JSON array detected.")
                                
                                json_candidate = response_text[start_index:end_index+1].strip()

                                # Attempt final JSON parse
                                try:
                                    enriched_blocks = json.loads(json_candidate)
                                    if not isinstance(enriched_blocks, list):
                                        raise ValueError("The JSON response is not an array.")
                                    rust_files_json.extend(enriched_blocks)
                                except json.JSONDecodeError as e:
                                    print("Error while parsing JSON:", e)
                                    print("Raw response after cleaning:", json_candidate)
                                    raise e

                            except Exception as e:
                                print(f"Error with OpenAI API: {e}")

                            file_count += 1
                            progress_bar.update(1)

            except requests.exceptions.RequestException as e:
                print(f"Error retrieving files for {repo_name}: {e}")

    progress_bar.close()

    # Conversion to DataFrame
    rust_scraping_df = pl.DataFrame({"json_data": rust_files_json})
    return rust_scraping_df

def save_dataframes_to_json(df_code_explanation, df_qa_final, rust_scraping_df, df_augmentation, output_dir="output"):
    """
    Saves Polar DataFrames as JSON (NDJSON) files.
    """
    os.makedirs(output_dir, exist_ok=True)

    def save_ndjson(df, path):
        with open(path, "w") as f:
            f.write("[\n")
            for i, row in enumerate(df.iter_rows()):
                json_data = row[0]
                f.write(json.dumps(json_data))
                if i < len(df) - 1:
                    f.write(",\n")
            f.write("\n]")

    save_ndjson(df_code_explanation, os.path.join(output_dir, "df_code_explanation.json"))
    save_ndjson(df_qa_final, os.path.join(output_dir, "df_qa_final.json"))
    save_ndjson(rust_scraping_df, os.path.join(output_dir, "df_rust_scraping.json"))
    save_ndjson(df_augmentation, os.path.join(output_dir, "df_augmentation.json"))

    print(f"All JSON files were saved in: {output_dir}")

def merge_json_files(output_dir="output", output_file="data_code.json"):
    """
    Merges the JSON files df_code_explanation.json, df_rust_scraping.json,
    and df_augmentation.json into a single JSON file structured as a list of dictionaries.
    Before writing the final file, removes all objects containing at least one field equal to None.
    Then removes duplicates.

    Args:
    output_dir (str): Directory containing the JSON files.
    output_file (str): Name of the merged JSON file.
    """
    files_to_merge = [
        os.path.join(output_dir, "df_code_explanation.json"),
        os.path.join(output_dir, "df_rust_scraping.json"),
        os.path.join(output_dir, "df_augmentation.json"),
    ]

    merged_data = []

    for file_path in files_to_merge:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # Extract dictionaries directly
                for item in data:
                    if isinstance(item, dict):
                        # Check if "json_data" is present and is a dict
                        if "json_data" in item and isinstance(item["json_data"], dict):
                            merged_data.append(item["json_data"])
                        else:
                            merged_data.append(item)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for file {file_path} : {e}")

    # Filter elements with at least one None field
    filtered_data = []
    null_fields_count = 0
    for doc in merged_data:
        if all(value is not None for value in doc.values()):
            filtered_data.append(doc)
        else:
            null_fields_count += 1

    # Remove duplicates
    # We use a set to memorize the objects already encountered.
    # We transform each doc into a JSON string sorted by keys, in order to identify duplicates.
    unique_data = []
    seen = set()
    duplicates_count = 0

    for doc in filtered_data:
        doc_str = json.dumps(doc, sort_keys=True)
        if doc_str not in seen:
            seen.add(doc_str)
            unique_data.append(doc)
        else:
            duplicates_count += 1

    # Save filtered and unique data in the final file
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(unique_data, f, indent=4)

    print(f"Merged JSON file successfully created and filtered: {output_path}")
    print(f"Number of deleted documents having at least one None field: {null_fields_count}")
    print(f"Number of duplicate documents deleted: {duplicates_count}")

def main():
    """
    Runs the full pipeline.
    """
    # Step 1: Get the Rust datasets
    df_code_explanation, df_qa_final = fetch_and_process_datasets()
    print("DataFrame Code Explanation (JSON):")
    print(df_code_explanation)
    print("\nDataFrame QA Final (JSON):")
    print(df_qa_final)

    # Step 2: Scrape Rust files from GitHub
    rust_scraping_df = scrape_github_rust_files("rust-lang")
    print("\nRust Scraping DataFrame:")
    print(rust_scraping_df)
    
    
    # Step 3: Generate augmented data
    df_augmentation = generate_augmented_data_with_openai()
    print("\nDataFrame Augmentation (JSON):")
    print(df_augmentation)

    
    # Step 4: Save all DataFrames as JSON
    save_dataframes_to_json(df_code_explanation, df_qa_final, rust_scraping_df, df_augmentation)
    
    
    # Step 5: Merge JSON files to create data_code.json
    merge_json_files()

    
if __name__ == "__main__":
    main()
