# Ensure the Procfile exists with the following line:
# web: gunicorn app:app

from flask import Flask, request, jsonify
from flask_cors import CORS
from llmproxy import generate, pdf_upload
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.INFO)

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

@app.route('/', methods=['GET', 'POST'])
def root_handler():
    if request.method == 'POST':
        data = request.get_json() or {}
        logging.info(f"Received POST request at root: {data}")

        # Provide instructions if no question is found
        parent_question = data.get("text", "").strip()
        if not parent_question:
            return jsonify({
                "message": "Welcome! To get started, you can ask questions like:\n"
                           "- How can I support my child with autism at home?\n"
                           "- What activities can help with learning at home?\n"
                           "- How do I handle sensory sensitivities?\n\n"
                           "For IEP generation, please provide:\n"
                           "- Student's Name\n"
                           "- Year of Education (e.g., Grade 3, Year 5)\n"
                           "- School Location (city, state, or country)."
            })

        logging.info(f"Processing question from root POST: {parent_question}")
        qna_prompt = """
        You are a compassionate and knowledgeable advisor who supports parents of children with autism. 
        Provide clear, empathetic, and actionable responses to general questions from parents. 
        Do not provide medical diagnoses or opinions; instead, focus on practical advice, educational strategies, and emotional support tips.
        """

        try:
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
            return jsonify({"success": True, "response": response["response"]})
        except Exception as e:
            logging.error(f"Error processing question at root: {e}")
            return jsonify({"error": f"Failed to process question: {str(e)}"}), 500

    return jsonify({"text": "Hello from Koyeb - you reached the main page for IEP Generator & Parent Q&A!\n"
                             "To generate an IEP, provide:\n"
                             "- Student's Name\n"
                             "- Year of Education (e.g., Grade 3, Year 5)\n"
                             "- School Location (city, state, or country).\n\n"
                             "Ask general questions like:\n"
                             "- How can I support my child with autism at home?\n"
                             "- What activities help with learning at home?"})

@app.route('/generate_iep', methods=['POST'])
def generate_iep():
    try:
        data = request.get_json()
        logging.info(f"Received generate_iep request: {data}")

        student_name = data.get("student_name", "Unknown")
        education_year = data.get("education_year", "")
        school_location = data.get("school_location", "")

        if not student_name or not education_year or not school_location:
            return jsonify({
                "error": "Missing required fields. To generate an IEP, provide:\n"
                         "- Student's Name\n"
                         "- Year of Education (e.g., Grade 3, Year 5)\n"
                         "- School Location (city, state, or country)."
            }), 400

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

        return jsonify({"success": True, "iep": response["response"]})

    except Exception as e:
        logging.error(f"Error generating IEP: {e}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/parent_qna', methods=['POST'])
def parent_qna():
    try:
        data = request.get_json()
        logging.info(f"Received parent_qna request: {data}")

        parent_question = data.get("question", "")
        if not parent_question.strip():
            return jsonify({
                "error": "Please provide a valid question. For example:\n"
                         "- How can I support my child with autism at home?\n"
                         "- What activities can help with learning at home?"
            }), 400

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

        return jsonify({"success": True, "response": response["response"]})

    except Exception as e:
        logging.error(f"Error in parent_qna: {e}")
        return jsonify({"error": f"Q&A failed: {str(e)}"}), 500

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Not Found"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)


