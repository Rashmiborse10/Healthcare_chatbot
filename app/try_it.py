from dotenv import load_dotenv
load_dotenv()

from app.llm.healthcare_assistant import HealthcareAssistant

assistant = HealthcareAssistant()
result = assistant.answer("what is Modern Cardiac Diagnostics?")
print(result.answer)
#print(result.citations)