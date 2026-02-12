import shap
import numpy as np
import pandas as pd

class UniversalSemanticReasoner:
    def __init__(self, pipeline, background_data, semantic_config):
        """
        A Universal Neuro-Symbolic Reasoner that explains ANY model (Classification OR Regression).
        
        Args:
            pipeline: A trained sklearn Pipeline (must have 'preprocessor' and 'model').
            background_data: A sample of processed X_train (numpy array or DataFrame) for SHAP baseline.
            semantic_config: Dictionary mapping feature groups to human meanings.
        """
        self.pipeline = pipeline
        self.config = semantic_config
        
        # safely extract steps
        self.preprocessor = pipeline.named_steps['preprocessor']
        self.model = pipeline.named_steps['model']
        
        # --- 1. TASK DETECTION (The Universal Switch) ---
        if hasattr(self.model, "predict_proba"):
            self.task_type = "classification"
            print("Detected Task: CLASSIFICATION (using predict_proba)")
        else:
            self.task_type = "regression"
            print("Detected Task: REGRESSION (using predict)")
        
        # Initialize SHAP Explainer
        print("Initializing Semantic Reasoner... (this may take a moment)")
        
        # Handle Data Type (Pandas vs Numpy)
        if isinstance(background_data, pd.DataFrame):
             self.background = background_data.values
        else:
             self.background = background_data
             
        self.explainer = shap.Explainer(self.model, self.background)

    def _get_relative_strength(self, impact, total_impact):
        """
        Calculates strength based on % contribution to the decision.
        Works for any scale ($$$ or Probability).
        """
        if total_impact == 0: return "slightly"
        
        # Calculate percentage share of the total decision
        share = abs(impact) / total_impact
        
        if share > 0.20: return "strongly"     # Feature drove >20% of the shift
        if share > 0.10: return "moderately"   # Feature drove >10% of the shift
        return "slightly"

    def _aggregate_impacts(self, feature_names, shap_values):
        """Groups technical features into Semantic Concepts."""
        grouped = {k: 0.0 for k in self.config.keys()}
        grouped["Unknown Factors"] = 0.0
        
        for name, impact in zip(feature_names, shap_values):
            found = False
            for key in self.config.keys():
                # Flexible matching (case-insensitive)
                if key.lower() in name.lower():
                    grouped[key] += impact
                    found = True
                    break
            
            if not found:
                grouped["Unknown Factors"] += impact
        return grouped

    def explain(self, raw_input_df):
        """Generates Explanation for a single input row."""
        
        # 1. Pipeline Transformation
        processed_data = self.preprocessor.transform(raw_input_df)
        
        # --- 2. UNIVERSAL PREDICTION LOGIC ---
        if self.task_type == "classification":
            # Get probability of the positive class (1)
            pred_val = self.model.predict_proba(processed_data)[0][1]
            outcome = "Positive" if pred_val > 0.5 else "Negative"
        else:
            # Get raw continuous value (e.g., Price, Score)
            pred_val = self.model.predict(processed_data)[0]
            outcome = f"Value: {pred_val:.2f}"

        # 3. Attribution
        shap_values = self.explainer(processed_data)
        
        # Handle SHAP output shapes
        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if len(vals.shape) > 1 and vals.shape[1] == 2:
                # Binary classification often returns (rows, 2) -> take class 1
                raw_impacts = vals[0][:, 1] 
            elif len(vals.shape) > 1:
                 # Multiclass or other shape
                 raw_impacts = vals[0]
            else:
                # Regression (rows, features)
                raw_impacts = vals[0]
        else:
            # Fallback for older SHAP versions
            raw_impacts = shap_values[0]

        feature_names = self.preprocessor.get_feature_names_out()

        # 4. Semantic Reasoning
        grouped = self._aggregate_impacts(feature_names, raw_impacts)
        
        # Calculate Total "Force" applied by features
        total_force = np.sum(np.abs(list(grouped.values()))) + 1e-9
        
        reasoning_log = []
        sorted_groups = sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for group, impact in sorted_groups:
            if abs(impact) < (total_force * 0.05): continue # Skip noise (<5% impact)
            
            direction = "increased" if impact > 0 else "decreased"
            strength = self._get_relative_strength(impact, total_force)
            
            if group == "Unknown Factors":
                meaning = "undefined features in the knowledge base"
            else:
                meaning = self.config[group]['meaning']
            
            reasoning_log.append(
                f"{group} {strength} {direction} the prediction, consistent with {meaning}."
            )

        # 5. Confidence Score
        # (How much of the decision did we successfully explain?)
        confidence = sum(abs(i) for g, i in sorted_groups if g != "Unknown Factors") / total_force
        conf_label = "HIGH" if confidence > 0.75 else "LOW"

        return {
            "prediction": round(float(pred_val), 4),
            "outcome": outcome,
            "reasoning": reasoning_log[:3],
            "confidence": conf_label
        }