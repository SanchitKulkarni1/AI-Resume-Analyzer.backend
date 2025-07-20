# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# import os
# import json
# import re # <== STEP 1: Import the regular expression library

# # Import your custom modules
# from extract_embed import extract_text
# from parsing_summary import analyze_resume, parse_resume
# from suggestion import suggest_resume_improvements
# from roadmap import generate_roadmap

# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# @app.route("/")
# def home():
#     return jsonify({"message": "Resume Analyzer API is running!"})

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     if 'resume' not in request.files:
#         return jsonify({"error": "No resume file uploaded"}), 400

#     file = request.files['resume']
#     job_description = request.form.get('job_description', '').strip()

#     if not file or file.filename == '':
#         return jsonify({"error": "No resume file uploaded"}), 400

#     if not job_description:
#         return jsonify({"error": "Job description is required"}), 400

#     try:
#         resume_text = extract_text(file)

#         # --- THIS IS THE CORRECTED LOGIC ---

#         # Call all your modules
#         raw_parsed_string = parse_resume(resume_text)
#         analysis_results_dict = analyze_resume(resume_text, job_description)
#         suggestions_string = suggest_resume_improvements(resume_text)

#         score = analysis_results_dict.get("score")
#         analysis_text = analysis_results_dict.get("analysis")
        
#         # ==> STEP 2: Clean the raw string from the parser
#         # Find the JSON object within the potentially messy string
#         match = re.search(r"\{.*\}", raw_parsed_string, re.DOTALL)
#         if not match:
#             # If no JSON is found at all, return an error
#             return jsonify({"error": "Failed to parse resume content from LLM output"}), 500
        
#         # Extract the clean JSON string
#         clean_json_string = match.group(0)
        
#         # Now, safely load the clean string
#         parsed_data = json.loads(clean_json_string)

#         roadmap_string = generate_roadmap(parsed_data, analysis_text, job_description, score)

#         # Assemble the final response
#         final_response = {
#             "parsed": parsed_data,
#             "analysis": analysis_text,
#             "score": score,
#             "suggestions": suggestions_string,
#             "roadmap": roadmap_string
#         }
        
#         return jsonify(final_response)

#     except json.JSONDecodeError:
#         # Catch the specific error if the cleaned string is still not valid JSON
#         return jsonify({"error": "LLM returned malformed JSON."}), 500
#     except Exception as e:
#         print(f"An error occurred in /analyze: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)