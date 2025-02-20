from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload

app = Flask(__name__)

prompt = """
You are a special needs academic advisor specializing in supporting young children with developmental learning disabilities, such as autism, dyslexia, and ADHD.

Your primary task is to create a **personalized Individualized Education Plan (IEP)** specifically designed for parents. 

The IEP should:
- Only take context from the document uploaded by the user.
- Provide actionable suggestions for parents on how to support their child's learning and development at home.
- Offer home-based strategies, routines, and activities tailored to the child's unique needs.
- Avoid any form of medical diagnosis or recommendations beyond home support.

No need to go over current strengths and weaknesses, as these are covered in the uploaded document.

**Important:**  
🔹 **Focus solely on creating a home-support educational plan for parents.**  
🔹 **Do not answer any questions that are unrelated to this task.**  
"""

MODEL_NAME = "4o-mini"
SESSION_ID = "IEP_1"
TEMPERATURE = 0.0
LASTK = 0
RAG_USAGE = True
RAG_THRESHOLD = 0.6
RAG_K = 5

@app.route('/')
def hello_world():
    return jsonify({"text": 'Hello from Koyeb - you reached the main page for IEP Generator & Parent Q&A!'})

@app.route('/generate_iep', methods=['POST'])
def generate_iep():
    data = request.get_json()
    student_name = data.get("student_name", "Unknown")
    education_year = data.get("education_year", "")
    school_location = data.get("school_location", "")
    
    if not student_name or not education_year or not school_location:
        return jsonify({"error": "Missing required fields: student_name, education_year, or school_location."}), 400

    query = (
        f"Using the student report uploaded for {student_name}, who is in {education_year} at a school in {school_location}, "
        "generate a detailed Individualized Education Plan (IEP) tailored to their educational needs for home support."
    )

    response = generate(
        model=MODEL_NAME,
        system=prompt,
        query=query,
        temperature=TEMPERATURE,
        lastk=LASTK,
        session_id=SESSION_ID,
        rag_usage=RAG_USAGE,
        rag_threshold=RAG_THRESHOLD,
        rag_k=RAG_K
    )

    return jsonify({"iep": response["response"]})

@app.route('/parent_qna', methods=['POST'])
def parent_qna():
    data = request.get_json()
    parent_question = data.get("question", "")

    if not parent_question.strip():
        return jsonify({"error": "Please provide a valid question."}), 400

    qna_prompt = """
    You are a compassionate and knowledgeable advisor who supports parents of children with autism. 
    Provide clear, empathetic, and actionable responses to general questions from parents. 
    Do not provide medical diagnoses or opinions; instead, focus on practical advice, educational strategies, and emotional support tips.
    """

    response = generate(
        model=MODEL_NAME,
        system=qna_prompt,
        query=parent_question,
        temperature=TEMPERATURE,
        lastk=LASTK,
        session_id=SESSION_ID,
        rag_usage=RAG_USAGE,
        rag_threshold=RAG_THRESHOLD,
        rag_k=RAG_K
    )

    return jsonify({"response": response["response"]})

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Not Found"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
