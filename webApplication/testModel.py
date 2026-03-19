"""
Script to test the downloaded model
"""

import torch
import os
from transformers import DistilBertTokenizer
import config
from model import DistilBertClassifier

def test_model():
    """Test the loaded model with sample text"""
    
    print("🧪 Testing model...")
    
    # Check if model exists
    model_path = f"{config.MODEL_SAVE_PATH}/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please download your trained model first.")
        return False
    
    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Load model
        print("📥 Loading model...")
        model = DistilBertClassifier()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Test with sample texts
        test_texts = [
            "The United Nations has called for immediate action on climate change.",
            "The team won the championship after an incredible comeback.",
            "The company reported record profits this quarter.",
            "Scientists have developed a new artificial intelligence algorithm."
        ]
        
        print("\n🔍 Testing with sample texts:")
        print("-" * 50)
        
        for i, text in enumerate(test_texts, 1):
            # Tokenize
            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=config.MAX_LENGTH,
                return_tensors='pt'
            )
            
            # Predict
            with torch.no_grad():
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            print(f"Text {i}: {text[:50]}...")
            print(f"Predicted: {config.CLASS_NAMES[predicted_class]} ({confidence:.1%})")
            print()
        
        print("✅ Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers',
        'streamlit',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def main():
    """Main function"""
    print("🚀 Model Testing Script")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if deps_ok:
        # Test model
        model_ok = test_model()
        
        if model_ok:
            print("\n🎉 Everything is ready!")
            print("You can now run: streamlit run app.py")
        else:
            print("\n❌ Model test failed. Please check your model file.")
    else:
        print("\n❌ Dependencies missing. Please install them first.")

if __name__ == "__main__":
    main()

