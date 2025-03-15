"""
Main Execution Script for EpiCRISPR-UQ
"""
from model import EpiCRISPR_UQ
from data_processor import DataProcessor
from trainer import ModelTrainer
from evaluator import ModelEvaluator
import numpy as np
import pandas as pd
import logging
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_data():
    """Load CRISPR dataset"""
    logger.info("Loading CRISPRv2 dataset")
    filepath = "C:\\Users\\desti\\Downloads\\CRISPRv2DATASET.csv"
    df = pd.read_csv(filepath, low_memory=False)
    print("Actual columns in dataset:", df.columns.tolist())
    logger.info(f"Loaded dataframe with shape: {df.shape}")
    return df

def setup_output_directory():
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {os.path.abspath(output_dir)}")
    return output_dir

def save_results(metrics, output_dir):
    pd.DataFrame([metrics]).to_csv(f'{output_dir}/metrics.csv', index=False)
    if 'history' in metrics:
        pd.DataFrame(metrics['history'].history).to_csv(
            f'{output_dir}/training_history.csv', index=False)
    logger.info("Results saved successfully")

def main():
    output_dir = setup_output_directory()
    logger.info("Starting EpiCRISPR-UQ pipeline")
    
    try:
        # Load and examine data
        df = load_real_data()
        
        # Initialize components
        model = EpiCRISPR_UQ(sequence_length=40, n_epigenetic_features=7)
        processor = DataProcessor(sequence_length=40)
        trainer = ModelTrainer(model)
        evaluator = ModelEvaluator(model)
        
        # Process data
        logger.info("Processing sequence and epigenetic data")
        X, y = processor.process(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        logger.info("Training model")
        history = trainer.train(X_train, y_train)
        
        # Evaluate
        logger.info("Evaluating model performance")
        metrics = evaluator.evaluate(X_test, y_test)
        save_results({'history': history, **metrics}, output_dir)
            
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
