from bs4 import BeautifulSoup
import requests
import time
import json

MAX_REQUESTS = 250
request_count = 0

def clean_html(html_content):
    """Cette fonction nettoie le contenu HTML et retourne le texte brut."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def fetch_questions_with_accepted_answers_and_content(tag, output_file):
    base_url = "https://api.stackexchange.com/2.3/questions"
    params = {
        "site": "stackoverflow",
        "tagged": tag,
        "hasaccepted": "true",
        "pagesize": 100,
        "page": 1,
        "filter": "withbody"
    }
    all_questions = []
    while True:
        print(f"Fetching page {params['page']}...")
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        data = response.json()
        all_questions.extend(data["items"])
        if not data["has_more"]:
            break
        params["page"] += 1
        time.sleep(1)  # Respecter les limites de l'API

    # Création d'un dictionnaire avec toutes les questions et leurs contenus nettoyés
    cleaned_data = []
    for question in all_questions:
        body_cleaned = clean_html(question.get("body", "No body content available"))
        cleaned_data.append({
            "question_id": question["question_id"],
            "title": question["title"],
            "body": body_cleaned
        })

    # Écrire les données nettoyées dans un fichier JSON
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(cleaned_data, json_file, ensure_ascii=False, indent=4)
    print(f"Saved {len(all_questions)} questions with content to {output_file}.")


# Fonction pour récupérer la réponse acceptée d'une question
def get_accepted_answer_id(question_id):
    url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
    params = {
        "order": "desc",
        "sort": "activity",
        "site": "stackoverflow",
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        answers = response.json().get("items", [])
        # Trouver la réponse marquée comme acceptée
        for answer in answers:
            if answer.get("is_accepted", False):
                return answer.get("answer_id", None)
        return None
    except Exception as e:
        print(f"Erreur lors de la récupération de la réponse pour la question {question_id}: {e}")
        return None
    

def get_accepted_answer_body(answer_id):
    url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
    params = {
        "order": "desc",
        "sort": "activity",
        "site": "stackoverflow",
        "filter": "withbody",
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        answer = response.json().get("items", [])[0]
        return answer.get("body", "Aucun corps de réponse trouvé")
    except Exception as e:
        print(f"Erreur lors de la récupération du corps de la réponse {answer_id}: {e}")
        return "Erreur lors de la récupération"


if __name__ == "__main__":
    fetch_questions_with_accepted_answers_and_content(tag="rust", output_file="question_data.json")

        # Charger le fichier JSON contenant les questions
    with open("question_data.json", "r", encoding="utf-8") as json_file:
        questions_data = json.load(json_file)

    for question in questions_data:
        if request_count >= MAX_REQUESTS:
            print("Limite atteinte. Relancez le script pour continuer.")
            break
        question_id = question["question_id"]
        print(f"Récupération de la réponse acceptée pour la question {question_id}...")
        answer_id = get_accepted_answer_id(question_id)
        if answer_id:
            question["accepted_answer"] = get_accepted_answer_body(answer_id)
        else:
            question["accepted_answer"] = "Aucune réponse acceptée trouvée"
        request_count += 1

    # Sauvegarder les données enrichies dans un nouveau fichier JSON
    with open("questions_data_with_answers.json", "w", encoding="utf-8") as json_file:
        json.dump(questions_data, json_file, ensure_ascii=False, indent=4)

    print("Les données enrichies ont été enregistrées dans 'questions_data_with_answers.json'.")
