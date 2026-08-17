# app.py
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import json
import pickle
import os
from werkzeug.utils import secure_filename
from pathlib import Path

from src.utils.logger import logger
from src.config import (
    RAW_DATA_DIR, MODELS_DIR, TARGET_COLUMN,
    REQUIRED_COLUMNS, RISK_THRESHOLDS,
    PREPROCESSOR_PATH, LABEL_ENCODER_PATH,
    FEATURE_SELECTOR_PATH, BEST_MODEL_PATH
)
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.features.selection import FeatureSelector
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.predict import Predictor
from src.models.explainability import ModelExplainer

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(RAW_DATA_DIR)

# Initialize components
data_loader = DataLoader()
data_validator = DataValidator()
data_preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()
feature_selector = FeatureSelector()
model_trainer = ModelTrainer()
model_evaluator = ModelEvaluator()
predictor = Predictor()
model_explainer = ModelExplainer()

# Global state
current_dataset = None
current_X = None
current_y = None
current_feature_names = None
selected_feature_names = None
trained_models = None
best_model_name = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'message': 'Student Performance Prediction API is running',
        'status': 'healthy'
    })

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    global current_dataset
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded',
                'data': None
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected',
                'data': None
            }), 400
        
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        
        current_dataset = data_loader.load_csv(filepath)
        validation_results = data_validator.validate_dataset(current_dataset)
        
        if not validation_results['valid']:
            return jsonify({
                'success': False,
                'message': 'Dataset validation failed: ' + '; '.join(validation_results['errors']),
                'data': {
                    'errors': validation_results['errors'],
                    'warnings': validation_results['warnings']
                }
            }), 400
        
        data_info = data_loader.get_data_info()
        
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'data': {
                'rows': len(current_dataset),
                'columns': len(current_dataset.columns),
                'column_names': list(current_dataset.columns),
                'missing_values': data_info.get('missing_values', {}),
                'target_distribution': data_info.get('target_distribution', {}),
                'validation': validation_results
            }
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

@app.route('/api/dataset/summary', methods=['GET'])
def get_dataset_summary():
    global current_dataset
    
    if current_dataset is None:
        return jsonify({
            'success': False,
            'message': 'No dataset loaded',
            'data': None
        }), 400
    
    try:
        data_info = data_loader.get_data_info()
        preview = current_dataset.head(10).to_dict('records')
        
        return jsonify({
            'success': True,
            'message': 'Dataset summary retrieved',
            'data': {
                'total_rows': len(current_dataset),
                'total_columns': len(current_dataset.columns),
                'column_names': list(current_dataset.columns),
                'missing_values': data_info.get('missing_values', {}),
                'dtypes': data_info.get('dtypes', {}),
                'target_distribution': data_info.get('target_distribution', {}),
                'preview': preview
            }
        })
        
    except Exception as e:
        logger.error(f"Summary error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    global current_dataset, current_X, current_y, current_feature_names
    
    if current_dataset is None:
        return jsonify({
            'success': False,
            'message': 'No dataset loaded',
            'data': None
        }), 400
    
    try:
        X, y = data_preprocessor.preprocess(current_dataset)
        current_X = X
        current_y = y
        current_feature_names = data_preprocessor.get_feature_names()
        
        # Save as .pkl files
        with open(PREPROCESSOR_PATH, 'wb') as f:
            pickle.dump(data_preprocessor.preprocessor, f)
        
        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(data_preprocessor.label_encoder, f)
        
        return jsonify({
            'success': True,
            'message': 'Data preprocessed successfully',
            'data': {
                'features_shape': X.shape,
                'feature_names': current_feature_names[:10],
                'target_distribution': pd.Series(y).value_counts().to_dict()
            }
        })
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

@app.route('/api/features/engineer', methods=['POST'])
def engineer_features():
    global current_dataset
    
    if current_dataset is None:
        return jsonify({
            'success': False,
            'message': 'No dataset loaded',
            'data': None
        }), 400
    
    try:
        engineered_data = feature_engineer.engineer_features(current_dataset)
        current_dataset = engineered_data
        
        return jsonify({
            'success': True,
            'message': 'Feature engineering completed',
            'data': {
                'engineered_features': feature_engineer.get_engineered_feature_names(),
                'total_columns': len(engineered_data.columns)
            }
        })
        
    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

# @app.route('/api/features/importance', methods=['GET'])
# def get_feature_importance():
#     global current_feature_names, current_X, current_y, model_trainer
    
#     if current_X is None or current_y is None:
#         return jsonify({
#             'success': False,
#             'message': 'No processed data available',
#             'data': None
#         }), 400
    
#     try:
#         if model_trainer.best_model is not None:
#             model_explainer.set_model(model_trainer.best_model)
#             model_explainer.set_feature_names(current_feature_names)
            
#             importances = model_explainer.get_feature_importance()
#             top_features = model_explainer.get_top_features(10)
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Feature importance retrieved',
#                 'data': {
#                     'feature_importance': top_features,
#                     'all_importances': importances,
#                     'total_features': len(current_feature_names),
#                     'feature_names': current_feature_names
#                 }
#             })
#         else:
#             from sklearn.feature_selection import mutual_info_classif
#             mi_scores = mutual_info_classif(current_X, current_y, random_state=42)
            
#             if len(mi_scores) == len(current_feature_names):
#                 importances = dict(zip(current_feature_names, mi_scores))
#             else:
#                 importances = {f'feature_{i}': score for i, score in enumerate(mi_scores)}
            
#             sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
#             top_features = list(sorted_importances.items())[:10]
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Feature importance calculated using mutual information',
#                 'data': {
#                     'feature_importance': [{'feature': name, 'importance': float(score)} for name, score in top_features],
#                     'all_importances': sorted_importances,
#                     'total_features': len(current_feature_names),
#                     'feature_names': current_feature_names
#                 }
#             })
            
#     except Exception as e:
#         logger.error(f"Feature importance error: {e}")
#         return jsonify({
#             'success': False,
#             'message': str(e),
#             'data': None
#         }), 500

# app.py - Only the updated feature importance section

# @app.route('/api/features/importance', methods=['GET'])
# def get_feature_importance():
#     """Get feature importance with proper feature names"""
#     global current_feature_names, current_X, current_y, model_trainer
    
#     if current_X is None or current_y is None:
#         return jsonify({
#             'success': False,
#             'message': 'No processed data available',
#             'data': None
#         }), 400
    
#     try:
#         # Log current state for debugging
#         logger.info(f"=== FEATURE IMPORTANCE DEBUG ===")
#         logger.info(f"Current feature names count: {len(current_feature_names) if current_feature_names else 0}")
#         if current_feature_names:
#             logger.info(f"Current feature names: {current_feature_names}")
        
#         # Get feature importance from the best model
#         if model_trainer.best_model is not None:
#             # Set feature names on the explainer
#             model_explainer.set_model(model_trainer.best_model)
#             model_explainer.set_feature_names(current_feature_names)
            
#             importances = model_explainer.get_feature_importance()
#             top_features = model_explainer.get_top_features(10)
            
#             # Log for debugging
#             logger.info(f"Feature importance retrieved for {len(importances)} features")
#             if top_features:
#                 logger.info(f"Top features with names: {[f['feature'] for f in top_features]}")
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Feature importance retrieved',
#                 'data': {
#                     'feature_importance': top_features,
#                     'all_importances': importances,
#                     'total_features': len(current_feature_names),
#                     'feature_names': current_feature_names
#                 }
#             })
#         else:
#             # Calculate mutual information if model not trained
#             from sklearn.feature_selection import mutual_info_classif
#             mi_scores = mutual_info_classif(current_X, current_y, random_state=42)
            
#             # Use actual feature names
#             if current_feature_names and len(current_feature_names) == len(mi_scores):
#                 importances = dict(zip(current_feature_names, mi_scores))
#                 logger.info(f"Using actual feature names for MI: {list(current_feature_names)[:5]}")
#             else:
#                 importances = {f'Feature_{i+1}': score for i, score in enumerate(mi_scores)}
#                 logger.warning(f"Feature names count mismatch. Using generic names.")
            
#             sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
#             top_features = [{'feature': name, 'importance': float(score)} for name, score in list(sorted_importances.items())[:10]]
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Feature importance calculated using mutual information',
#                 'data': {
#                     'feature_importance': top_features,
#                     'all_importances': sorted_importances,
#                     'total_features': len(current_feature_names),
#                     'feature_names': current_feature_names
#                 }
#             })
            
#     except Exception as e:
#         logger.error(f"Feature importance error: {e}")
#         return jsonify({
#             'success': False,
#             'message': str(e),
#             'data': None
#         }), 500
# app.py - Updated feature importance route

@app.route('/api/features/importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance with proper feature names"""
    global current_feature_names, current_X, current_y, model_trainer, selected_feature_names
    
    if current_X is None or current_y is None:
        return jsonify({
            'success': False,
            'message': 'No processed data available',
            'data': None
        }), 400
    
    try:
        # Log current state for debugging
        logger.info(f"=== FEATURE IMPORTANCE DEBUG ===")
        logger.info(f"Selected feature names: {selected_feature_names}")
        logger.info(f"Current feature names count: {len(current_feature_names) if current_feature_names else 0}")
        
        # Determine which feature names to use
        # If feature selection was done, use selected feature names
        # Otherwise use all feature names
        if selected_feature_names is not None and len(selected_feature_names) > 0:
            feature_names_to_use = selected_feature_names
            logger.info(f"Using selected features ({len(feature_names_to_use)}): {feature_names_to_use}")
        else:
            feature_names_to_use = current_feature_names
            logger.info(f"Using all features ({len(feature_names_to_use) if feature_names_to_use else 0})")
        
        # Get feature importance from the best model
        if model_trainer.best_model is not None:
            # Set feature names on the explainer
            model_explainer.set_model(model_trainer.best_model)
            model_explainer.set_feature_names(feature_names_to_use)
            
            importances = model_explainer.get_feature_importance()
            top_features = model_explainer.get_top_features(10)
            
            # Log for debugging
            logger.info(f"Feature importance retrieved for {len(importances)} features")
            if top_features:
                logger.info(f"Top features with names: {[f['feature'] for f in top_features]}")
            
            return jsonify({
                'success': True,
                'message': 'Feature importance retrieved',
                'data': {
                    'feature_importance': top_features,
                    'all_importances': importances,
                    'total_features': len(feature_names_to_use) if feature_names_to_use else 0,
                    'feature_names': feature_names_to_use,
                    'selected_features': selected_feature_names
                }
            })
        else:
            # Calculate mutual information if model not trained
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(current_X, current_y, random_state=42)
            
            # Use actual feature names
            if feature_names_to_use and len(feature_names_to_use) == len(mi_scores):
                importances = dict(zip(feature_names_to_use, mi_scores))
                logger.info(f"Using actual feature names for MI: {list(feature_names_to_use)[:5]}")
            else:
                importances = {f'Feature_{i+1}': score for i, score in enumerate(mi_scores)}
                logger.warning(f"Feature names count mismatch. Using generic names.")
            
            sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            top_features = [{'feature': name, 'importance': float(score)} for name, score in list(sorted_importances.items())[:10]]
            
            return jsonify({
                'success': True,
                'message': 'Feature importance calculated using mutual information',
                'data': {
                    'feature_importance': top_features,
                    'all_importances': sorted_importances,
                    'total_features': len(feature_names_to_use) if feature_names_to_use else 0,
                    'feature_names': feature_names_to_use,
                    'selected_features': selected_feature_names
                }
            })
            
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500
    
@app.route('/api/features/select', methods=['POST'])
def select_features():
    global current_X, current_y, current_feature_names, selected_feature_names
    
    if current_X is None or current_y is None:
        return jsonify({
            'success': False,
            'message': 'No processed data available',
            'data': None
        }), 400
    
    try:
        data = request.get_json()
        n_features = data.get('n_features', 10)
        
        feature_selector.n_features = n_features
        X_selected, selected_features = feature_selector.select_features(
            current_X, current_y, current_feature_names
        )
        
        current_X = X_selected
        selected_feature_names = selected_features
        
        # Save feature selector as .pkl
        with open(FEATURE_SELECTOR_PATH, 'wb') as f:
            pickle.dump(feature_selector.selector, f)
        
        return jsonify({
            'success': True,
            'message': f'Selected top {len(selected_features)} features',
            'data': {
                'selected_features': selected_features,
                'original_count': len(current_feature_names),
                'selected_count': len(selected_features),
                'feature_scores': feature_selector.get_feature_importance()
            }
        })
        
    except Exception as e:
        logger.error(f"Feature selection error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

# @app.route('/api/train', methods=['POST'])
# def train_models():
#     global current_X, current_y, trained_models, best_model_name
    
#     if current_X is None or current_y is None:
#         return jsonify({
#             'success': False,
#             'message': 'No processed data available. Please run preprocessing first.',
#             'data': None
#         }), 400
    
#     try:
#         trained_models = model_trainer.train_all(current_X, current_y)
#         best_model_name = model_trainer.best_model_name
        
#         evaluation_results = model_evaluator.evaluate_all(
#             trained_models,
#             model_trainer.X_test,
#             model_trainer.y_test
#         )
        
#         model_evaluator.save_metrics()
#         comparison_table = model_evaluator.get_comparison_table()
        
#         # Save feature selector as .pkl if available
#         if feature_selector.selector is not None:
#             with open(FEATURE_SELECTOR_PATH, 'wb') as f:
#                 pickle.dump(feature_selector.selector, f)
#             logger.info("Feature selector saved as .pkl")
        
#         # Reload predictor with new model
#         predictor.load_models()
#         comparison_dict = comparison_table.to_dict('index')
        
#         return jsonify({
#             'success': True,
#             'message': 'Models trained successfully',
#             'data': {
#                 'best_model': best_model_name,
#                 'comparison_table': comparison_dict,
#                 'evaluation_results': evaluation_results,
#                 'models_trained': list(trained_models.keys())
#             }
#         })
        
#     except Exception as e:
#         logger.error(f"Training error: {e}")
#         return jsonify({
#             'success': False,
#             'message': str(e),
#             'data': None
#         }), 500
# app.py - Updated training route

@app.route('/api/train', methods=['POST'])
def train_models():
    global current_X, current_y, trained_models, best_model_name, selected_feature_names
    
    if current_X is None or current_y is None:
        return jsonify({
            'success': False,
            'message': 'No processed data available. Please run preprocessing first.',
            'data': None
        }), 400
    
    try:
        # Train models
        trained_models = model_trainer.train_all(current_X, current_y)
        best_model_name = model_trainer.best_model_name
        
        # Evaluate models
        evaluation_results = model_evaluator.evaluate_all(
            trained_models,
            model_trainer.X_test,
            model_trainer.y_test
        )
        
        model_evaluator.save_metrics()
        comparison_table = model_evaluator.get_comparison_table()
        
        # Save feature selector as .pkl if available
        if feature_selector.selector is not None:
            with open(FEATURE_SELECTOR_PATH, 'wb') as f:
                pickle.dump(feature_selector.selector, f)
            logger.info("Feature selector saved as .pkl")
        
        # IMPORTANT: Update the feature names to only include selected features
        if selected_feature_names is not None and len(selected_feature_names) > 0:
            # The model was trained on selected features
            # We need to update the explainer to use selected feature names
            logger.info(f"Training completed on {len(selected_feature_names)} selected features")
            logger.info(f"Selected features: {selected_feature_names}")
            
            # Update the model explainer with selected feature names
            model_explainer.set_feature_names(selected_feature_names)
        else:
            # If no feature selection was done, use all feature names
            if current_feature_names is not None:
                model_explainer.set_feature_names(current_feature_names)
        
        # Reload predictor with new model
        predictor.load_models()
        comparison_dict = comparison_table.to_dict('index')
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'data': {
                'best_model': best_model_name,
                'comparison_table': comparison_dict,
                'evaluation_results': evaluation_results,
                'models_trained': list(trained_models.keys()),
                'selected_features': selected_feature_names
            }
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

@app.route('/api/models/results', methods=['GET'])
def get_model_results():
    if not trained_models and not model_evaluator.metrics:
        return jsonify({
            'success': False,
            'message': 'No models trained. Please train models first.',
            'data': None
        }), 400
    
    try:
        comparison_table = model_evaluator.get_comparison_table()
        comparison_dict = comparison_table.to_dict('index')
        
        return jsonify({
            'success': True,
            'message': 'Model results retrieved',
            'data': {
                'comparison_table': comparison_dict,
                'best_model': best_model_name or model_trainer.best_model_name,
                'confusion_matrices': model_evaluator.confusion_matrices,
                'classification_reports': model_evaluator.classification_reports
            }
        })
        
    except Exception as e:
        logger.error(f"Model results error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided',
                'data': None
            }), 400
        
        if predictor.model is None or not predictor.is_loaded:
            predictor.load_models()
            if predictor.model is None:
                return jsonify({
                    'success': False,
                    'message': 'Model not trained. Please train models first.',
                    'data': None
                }), 400
        
        result = predictor.predict_single(data)
        
        return jsonify({
            'success': True,
            'message': 'Prediction completed successfully',
            'data': result
        })
        
    except ValueError as e:
        logger.error(f"Prediction value error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'message': f'Prediction failed: {str(e)}',
            'data': None
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded',
                'data': None
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected',
                'data': None
            }), 400
        
        df = pd.read_csv(file)
        
        if predictor.model is None or not predictor.is_loaded:
            predictor.load_models()
            if predictor.model is None:
                return jsonify({
                    'success': False,
                    'message': 'Model not trained. Please train models first.',
                    'data': None
                }), 400
        
        results = predictor.predict_batch(df)
        
        pred_counts = {}
        risk_counts = {}
        for r in results:
            pred_counts[r['prediction']] = pred_counts.get(r['prediction'], 0) + 1
            risk_counts[r['risk_level']] = risk_counts.get(r['risk_level'], 0) + 1
        
        return jsonify({
            'success': True,
            'message': 'Batch prediction completed',
            'data': {
                'results': results,
                'total': len(results),
                'predictions_summary': pred_counts,
                'risk_summary': risk_counts
            }
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

@app.route('/api/predict/download', methods=['POST'])
def download_predictions():
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({
                'success': False,
                'message': 'No results to download',
                'data': None
            }), 400
        
        df = pd.DataFrame(results)
        temp_path = Path('/tmp/predictions.csv')
        df.to_csv(temp_path, index=False)
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name='predictions.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

@app.route('/api/dataset/generate', methods=['POST'])
def generate_dataset():
    global current_dataset
    
    try:
        data = data_loader.load_synthetic_data()
        current_dataset = data
        
        return jsonify({
            'success': True,
            'message': 'Synthetic dataset generated successfully',
            'data': {
                'rows': len(data),
                'columns': len(data.columns),
                'target_distribution': data['performance_status'].value_counts().to_dict()
            }
        })
        
    except Exception as e:
        logger.error(f"Dataset generation error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)