# Pneumonia-Detector

## Configuration

The project uses Hydra for configuration management. Configuration files are located in the `configs` directory.

- **Model configurations**: `configs/model/`
- **Transform configurations**: `configs/transforms/`
- **General configuration**: `configs/config.yaml`

### Example configuration (`configs/config.yaml`):
```yaml
defaults:
  - model: simple
  - transforms: default

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  train_dir: 'data/train'
  val_dirs: ['data/val', 'data/test']
logging:
  steps: 10
  ```

## Training
To train the model, run the train.py script. You can specify different configurations through the command line.

Example command:
```bash
Copy code
python src/train.py training.learning_rate=0.005 model=moderate transforms=large
```

Streamlit Application
To visualize the confusion matrix for each saved checkpoint, use the Streamlit application.

Example command:
```bash
PYTHONPATH=. streamlit run dashboards/model_review.py
```
