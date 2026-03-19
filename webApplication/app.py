"""
Streamlit web application for DistilBERT text classification
Allows users to input text and get predictions
"""

import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer
import json
import os
from datetime import datetime

import config
from model import DistilBertClassifier

# Import constants from config
MAX_LENGTH = config.MAX_LENGTH
CLASS_NAMES = config.CLASS_NAMES

# Set page config
st.set_page_config(
    page_title="Transformer-based NLP Application",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Load model
        model = DistilBertClassifier()
        
        # Load trained weights
        model_path = f"{config.MODEL_SAVE_PATH}/best_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            st.error("Trained model not found. Please run training first.")
            return None, None
        
        model.eval()
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_text(text, model, tokenizer):
    """Predict the class of input text"""
    # Tokenize the text
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].numpy()

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🤖 Transformer-based NLP Application")
    st.subheader("Text Classification using DistilBERT")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    This application uses a fine-tuned DistilBERT model to classify news articles into four categories:
    
    - **World**: International news and events
    - **Sports**: Sports news and events  
    - **Business**: Business and financial news
    - **Sci/Tech**: Science and technology news
    
    The model was trained on the AG News dataset using the DistilBERT architecture.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model. Please ensure the model has been trained.")
        return
    
    st.success("Model loaded successfully!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Classification")
        
        # Text input
        text_input = st.text_area(
            "Enter your news text here:",
            height=200,
            placeholder="Paste your news article text here..."
        )
        
        # Example texts
        st.markdown("**Example texts to try:**")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            if st.button("World News Example"):
                st.session_state.example_text = "The United Nations has called for immediate action on climate change during the annual summit in New York."
            
            if st.button("Sports News Example"):
                st.session_state.example_text = "The team won the championship after an incredible comeback in the final minutes of the game."
        
        with example_col2:
            if st.button("Business News Example"):
                st.session_state.example_text = "The company reported record profits this quarter, exceeding analyst expectations by 15%."
            
            if st.button("Tech News Example"):
                st.session_state.example_text = "Scientists have developed a new artificial intelligence algorithm that can predict weather patterns with unprecedented accuracy."
        
        # Update text area if example is selected
        if hasattr(st.session_state, 'example_text'):
            text_input = st.session_state.example_text
            st.text_area("Enter your news text here:", value=text_input, height=200)
        
        # Predict button
        if st.button("🔍 Classify Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    # Get prediction
                    predicted_class, confidence, all_probabilities = predict_text(text_input, model, tokenizer)
                    
                    # Display results
                    st.markdown("### Results")
                    
                    # Show predicted category
                    st.metric(
                        label="Predicted Category",
                        value=CLASS_NAMES[predicted_class],
                        delta=f"{confidence:.1%} confidence"
                    )
                    
                    # Progress bars for all classes
                    st.markdown("**Confidence by Category:**")
                    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_probabilities)):
                        st.write(f"{class_name}: {prob:.1%}")
                        st.progress(prob)
                    
                    # Store prediction history
                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    st.session_state.prediction_history.append({
                        'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                        'predicted_class': CLASS_NAMES[predicted_class],
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
            else:
                st.warning("Please enter some text to classify.")
    
    with col2:
        st.header("📊 Model Info")
        
        # Model statistics
        st.metric("Model", "DistilBERT")
        st.metric("Classes", len(CLASS_NAMES))
        st.metric("Max Length", f"{MAX_LENGTH} tokens")
        
        # Show prediction history
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            st.markdown("### Recent Predictions")
            
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):
                with st.expander(f"{pred['predicted_class']} - {pred['timestamp']}"):
                    st.write(f"**Text:** {pred['text']}")
                    st.write(f"**Confidence:** {pred['confidence']:.1%}")
            
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Transformer-based NLP Application | Built with DistilBERT and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 