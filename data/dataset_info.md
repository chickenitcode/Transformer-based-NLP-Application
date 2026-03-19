# AG News Dataset Information

## Overview
The AG News dataset is a collection of news articles from 4 different categories:
- **World**: International news and events
- **Sports**: Sports news and events  
- **Business**: Business and financial news
- **Sci/Tech**: Science and technology news

## Dataset Structure
```
data/
├── raw/                    # Raw dataset files
│   ├── train.csv          # Training data
│   └── test.csv           # Test data
├── processed/              # Processed data
├── splits/                 # Train/val/test splits
├── sample/                 # Sample data for testing
│   ├── world_samples.txt
│   ├── sports_samples.txt
│   ├── business_samples.txt
│   ├── sci_tech_samples.txt
│   └── metadata.json
└── dataset_info.md         # This file
```

## Usage
- **Training**: Use files in `raw/` for training the model
- **Testing**: Use files in `sample/` for quick testing
- **Evaluation**: Use `test.csv` for final evaluation

## Class Distribution
- World: ~30,000 articles
- Sports: ~30,000 articles  
- Business: ~30,000 articles
- Sci/Tech: ~30,000 articles

## Format
Each line contains: `class_id,title,description`
- class_id: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
- title: Article title
- description: Article content
