# Ensure the Procfile exists with the following line:
# web: gunicorn app:app

from flask import Flask, request, jsonify, make_response
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

def create_json_response(data, status_code=200):
    response = make_response(jsonify(data), status_code)
    response.headers["Content-Type"] = "application/json"
    return response

@app.route('/', methods=['GET', 'POST'])
def root_handler():
    if request.method == 'POST':
        data = request.get_json() or {}
        logging.info(f"Received POST request at root: {data}")

        parent_question = data.get("text", "").strip()
        if not parent_question:
            return create_json_response({
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

        if parent_question.lower() == "how can i support my child with autism at home?":
            logging.info("Providing tailored response for common question.")
            response_data = {
                "success": True,
                "response": "Supporting your child with autism at home involves creating structured routines, offering clear communication, and providing sensory-friendly spaces. Consider introducing visual schedules, engaging in special interests together, and using positive reinforcement to encourage desired behaviors. Always provide a safe and predictable environment where your child can thrive."
            }
            logging.info(f"Returning tailored response: {response_data}")
            return create_json_response(response_data)

        try:
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

            response_data = {"success": True, "response": response["response"]}
            logging.info(f"Generated response to be returned: {response_data}")
            return create_json_response(response_data)
        except Exception as e:
            logging.error(f"Error processing question at root: {e}")
            return create_json_response({"error": f"Failed to process question: {str(e)}"}, 500)

    return create_json_response({"text": "Hello from Koyeb - you reached the main page for IEP Generator & Parent Q&A!\n"
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
            return create_json_response({
                "error": "Missing required fields. To generate an IEP, provide:\n"
                         "- Student's Name\n"
                         "- Year of Education (e.g., Grade 3, Year 5)\n"
                         "- School Location (city, state, or country)."
            }, 400)

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

        response_data = {"success": True, "iep": response["response"]}
        logging.info(f"Generated IEP response to be returned: {response_data}")
        return create_json_response(response_data)

    except Exception as e:
        logging.error(f"Error generating IEP: {e}")
        return create_json_response({"error": f"Generation failed: {str(e)}"}, 500)

@app.route('/parent_qna', methods=['POST'])
def parent_qna():
    try:
        data = request.get_json()
        logging.info(f"Received parent_qna request: {data}")

        parent_question = data.get("question", "")
        if not parent_question.strip():
            return create_json_response({
                "error": "Please provide a valid question. For example:\n"
                         "- How can I support my child with autism at home?\n"
                         "- What activities can help with learning at home?"
            }, 400)

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

        response_data = {"success": True, "response": response["response"]}
        logging.info(f"Generated Q&A response to be returned: {response_data}")
        return create_json_response(response_data)

    except Exception as e:
        logging.error(f"Error in parent_qna: {e}")
        return create_json_response({"error": f"Q&A failed: {str(e)}"}, 500)

@app.errorhandler(404)
def page_not_found(e):
    return create_json_response({"error": "Not Found"}, 404)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)



