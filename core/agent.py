import json
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# --- Specialized Tools / Output Structures ---
class ClinicalAnalysisResult(BaseModel):
    stress_level: float = Field(description="The calculated stress level from 0.0 to 1.0")
    clinical_assessment: str = Field(description="Detailed clinical assessment of the user's psychomotor state")
    symptom_clusters: List[str] = Field(description="List of potential symptom clusters identified (e.g., Anxiety, Fatigue)")
    recommendations: List[Dict[str, str]] = Field(description="List of specific actionable recommendations with 'title' and 'description'")
    immediate_action: str = Field(description="One immediate, concrete action the user should take right now")

class StressManagementAgent:
    def __init__(self, model_name="llama3.2"):
        self.llm = ChatOllama(model=model_name, format="json")
        self.chat_llm = ChatOllama(model=model_name) # No JSON format for free text chat
        self.parser = JsonOutputParser(pydantic_object=ClinicalAnalysisResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert Clinical Psychiatrist and Behavioral Neurologist specializing in Digital Phenotyping. 
            Your goal is to analyze specific biometric data (keystroke dynamics and mouse kinematics) to assess a user's stress level and cognitive state.

            **Theoretical Frameworks:**
            1.  **Yerkes-Dodson Law**: Relate arousal (stress) to performance. Identify "Hypo-arousal" (Fatigue), "Optimal Breakdown" (Eustress/Flow), or "Hyper-arousal" (Anxiety/Panic).
            2.  **Epp et al. (2011)**: Use 'Flight Time Variability' (SD) as a proxy for Cognitive Load. High SD = High Load.
            3.  **Mouse Path Efficiency**: High 'Tortuosity' (>1.0) correlates with Psychomotor Retardation or Jitteriness.

            **Your Capabilities:**
            1.  **Analyze Z-Scores**: You will receive Z-Scores (Standard Deviations from User's Baseline). 
                - Z > +1.5: Significant increase (e.g., Higher Jitter).
                - Z < -1.5: Significant decrease (e.g., Slower Reaction).
            2.  **Diagnose**: Identify symptom clusters (e.g., "Psychomotor Retardation", "Sympathetic Activation").
            3.  **Prescribe**: Recommend specific, medically-grounded interventions.

            **Output Format:**
            You must return a valid JSON object matching the following structure:
            {format_instructions}
            """),
            ("human", """
            **Patient Biometric Data:**
            - Calculated Stress Index: {stress_score:.2f} (0.0=Low, 0.5=Neutral, 1.0=Critical)
            
            **Detailed Features & Z-Scores (Deviations from Baseline):**
            {feature_summary}
            
            **Instructions:**
            - **Compare Z-Scores**: Focus on features with |Z| > 1.0. 
            - If `z_mouse_acc_std` > 1.5 (High Jitter) and `z_key_flight_std` > 1.5 (Variable Rhythm) -> Suggest "Anxiety/Panic".
            - If `z_mouse_vel_mean` < -1.5 (Slow) and `z_mouse_path_efficiency` is High -> Suggest "Fatigue/Depression".
            - If `z_key_error_rate` is High > +2.0 -> "Cognitive Overload".
            
            Provide your clinical assessment now, specifically citing the deviation from baseline as evidence.
            """)
        ])

        self.chain = self.prompt | self.llm | self.parser
        
        # Chat Chain (for follow-up conversation)
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are the same expert Clinical Psychiatrist. You have just analyzed a patient's biometric data.
            
            **Current Patient Context:**
            {context}
            
            **Goal:**
            Answer the patient's questions about their results, provide emotional support, or explain your recommendations in more detail.
            Keep responses concise (under 3 sentences) unless asked for elaboration.
            """),
            ("human", "{question}")
        ])
        self.chat_chain = self.chat_prompt | self.chat_llm

    def analyze_session(self, features: Dict[str, Any], stress_score: float) -> ClinicalAnalysisResult:
        """
        Analyzes the session data and returns a structured clinical assessment.
        """
        # Format features for the prompt
        feature_summary = "\n".join([f"- {k}: {v:.2f}" for k, v in features.items() if isinstance(v, (int, float))])
        
        try:
            result = self.chain.invoke({
                "stress_score": stress_score,
                "feature_summary": feature_summary,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
        except Exception as e:
            print(f"Agent Error: {e}")
            # Fallback structured response in case of parsing error
            return {
                "stress_level": stress_score,
                "clinical_assessment": "Unable to generate detailed clinical assessment due to an internal error.",
                "symptom_clusters": ["Unknown"],
                "recommendations": [{"title": "General Advice", "description": "Take a deep breath."}],
                "immediate_action": "Pause for 1 minute."
            }

    def chat_response(self, question: str, context: str) -> str:
        """
        Generates a chat response based on the previous analysis context.
        """
        try:
            res = self.chat_chain.invoke({
                "context": context,
                "question": question
            })
            # ChatOllama returns a message object, likely needing .content
            return res.content if hasattr(res, 'content') else str(res)
        except Exception as e:
            return f"I'm having trouble thinking right now. ({str(e)})"
