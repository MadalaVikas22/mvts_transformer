"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Custom packages
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, monitor='accuracy'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
        
def save_checkpoint(model, optimizer, epoch, metrics, config, filename='checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': time.time()
    }
    filepath = os.path.join(config['save_dir'], filename)
    torch.save(checkpoint, filepath)
    return filepath

def load_checkpoint(filepath, model, optimizer, device):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer


def main(config):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]

    # ADD THESE DEBUG PRINTS BEFORE DATA LOADING:
    print(f"Data directory: {config['data_dir']}")
    print(f"Pattern for training: {config['pattern']}")
    print(f"Pattern for validation: {config['val_pattern']}")
    print(f"Pattern for test: {config['test_pattern']}")

    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    # ADD MORE DEBUG PRINTS AFTER DATA LOADING:
    print(f"Total data loaded: {len(my_data.feature_df)} samples")
    if hasattr(my_data, 'train_IDs'):
        print(f"Train samples: {len(my_data.train_IDs)}")
    if hasattr(my_data, 'val_IDs'):
        print(f"Val samples: {len(my_data.val_IDs)}")
    if hasattr(my_data, 'test_IDs'):
        print(f"Test samples: {len(my_data.test_IDs)}")
    
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    if config['task'] == 'classification':
        # Get labels that match the actual loaded data indices
        if hasattr(my_data, 'labels_df'):
            # Make sure labels match the actual data indices
            available_indices = my_data.all_IDs
            labels = my_data.labels_df.loc[available_indices, 'target_binary'].values
            print(f"Data indices: {len(available_indices)}, Labels: {len(labels)}")
        else:
            labels = None
        validation_method = 'StratifiedShuffleSplit'
    else:
        labels = None
        validation_method = 'train_test_split'

    # Split dataset
    test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    if config['test_pattern']:  # used if test data come from different files / file patterns
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config['test_from']:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
        test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
        try:
            test_indices = [int(ind) for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
        logger.info("Loaded {} test IDs from file: '{}'".format(len(test_indices), config['test_from']))
    if config['val_pattern']:  # used if val data come from different files / file patterns
        val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs
        
        # ADD THIS DEBUG:
        print(f"Val data loaded: {len(val_data.feature_df)} samples")
        print(f"Val indices: {len(val_indices)} samples")

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    elif config['val_ratio'] > 0:
        # Use ratio-based splitting
        train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                validation_method=validation_method,
                                                                n_splits=1,
                                                                validation_ratio=config['val_ratio'],
                                                                test_set_ratio=config['test_ratio'],
                                                                test_indices=test_indices,
                                                                random_seed=1337,
                                                                labels=labels)
        train_indices = train_indices[0]
        val_indices = val_indices[0]
    else:
        # Default fallback: use 80/20 split
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            my_data.all_IDs, test_size=0.2, random_state=1337
        )
        test_indices = []

    # ADD MORE DEBUG:
    print(f"Final train_indices: {len(train_indices)}")
    print(f"Final val_indices: {len(val_indices)}")
    print(f"Final test_indices: {len(test_indices)}")

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices),
                       'test_indices': list(test_indices)}, f, indent=4)

    # Pre-process features
    # Pre-process features
    normalizer = None

    def normalize_subset(feature_df, indices, normalizer):
        """
        Normalizes specified indices of feature_df using the given normalizer
        """
        # Handle different types of indices
        if isinstance(indices, pd.Index):
            # If it's a pandas Index, use it directly
            subset = feature_df.loc[indices]
            normalized_subset = normalizer.normalize(subset)
            feature_df.loc[indices] = normalized_subset
        elif hasattr(indices, 'itertuples'):
            # If it's a DataFrame, convert to tuples as before
            index_tuples = list(indices.itertuples(index=False, name=None))
            subset = feature_df.loc[index_tuples]
            normalized_subset = normalizer.normalize(subset)
            feature_df.loc[index_tuples] = normalized_subset
        else:
            # If it's a list or array, use directly
            subset = feature_df.loc[indices]
            normalized_subset = normalizer.normalize(subset)
            feature_df.loc[indices] = normalized_subset


    # Load or create normalizer
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)

    elif config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])

        print(f"Type of train_indices: {type(train_indices)}")
        print(f"First 5 train_indices:\n{train_indices[:5]}")
        print(f"Index names in feature_df: {my_data.feature_df.index.names}")

        # Normalize training data
        normalize_subset(my_data.feature_df, train_indices, normalizer)

        # Save normalizing parameters for future use if not per-sample
        if not config['normalization'].startswith('per_sample'):
            norm_dict = normalizer.__dict__
            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)

    # Normalize validation and test data
    if normalizer is not None:
        normalize_subset(val_data.feature_df, val_indices, normalizer)
        normalize_subset(test_data.feature_df, test_indices, normalizer)


    loss_module = get_loss_module(config)

    if config['test_only'] == 'testset':  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                            print_interval=config['print_interval'], console=config['console'])
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        return
    
    # Initialize data generators FIRST
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    
    # Create datasets with label_column
    label_column = config.get('label_column', 'target_binary')  # Add default

    train_dataset = dataset_class(my_data, train_indices, label_column=label_column)
    val_dataset = dataset_class(val_data, val_indices, label_column=label_column)

    # Get the actual number of classes from your chosen label column
    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    print("Class distribution:")
    print(train_dataset.get_class_distribution())

    
    # NOW create the model with correct num_classes
    logger.info("Creating model ...")
    model = model_factory(config, my_data, num_classes=num_classes)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer AFTER model creation
    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    # Load model if specified
    start_epoch = 0
    lr_step = 0
    lr = config['lr']
    if config.get('load_model'):  # Changed from args.load_model
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'], config['lr'], config['lr_step'], config['lr_factor'])
    model.to(device)

    loss_module = get_loss_module(config)

    # Create data loaders with reduced batch size and no multiprocessing
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=8,  # Reduced batch size
                              shuffle=True,
                              num_workers=0,  # Disabled multiprocessing
                              pin_memory=False,  # Disabled for CPU
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=8,  # Reduced batch size
                            shuffle=False,
                            num_workers=0,  # Disabled multiprocessing
                            pin_memory=False,  # Disabled for CPU
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    # Initialize training history and early stopping
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': [],
        'epochs': []
    }
    
    early_stopping = EarlyStopping(
        patience=config.get('patience', 15),
        min_delta=config.get('min_delta', 0.001),
        monitor=config.get('monitor', 'accuracy')
    )

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16
    metrics = []
    best_metrics = {}
    best_model_path = None

    # Evaluate on validation before training
    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    # Continue with training loop...
    logger.info('Starting training...')
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        
        # Training
        aggr_metrics_train = trainer.train_epoch(epoch)
        epoch_runtime = time.time() - epoch_start_time
        
        # Log training metrics
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = aggr_metrics_train.get('loss', 0)
        train_accuracy = aggr_metrics_train.get('accuracy', 0)
        
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        tensorboard_writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        
        # Update training history
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['learning_rates'].append(current_lr)
        training_history['epochs'].append(epoch)
        
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # Validation
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))
            
            # Update validation history
            val_loss = aggr_metrics_val.get('loss', 0)
            val_accuracy = aggr_metrics_val.get('accuracy', 0)
            training_history['val_loss'].append(val_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            # Check if this is the best model
            current_metric = aggr_metrics_val.get(config['key_metric'], 0)
            is_best = False
            if config['key_metric'] in NEG_METRICS:
                is_best = current_metric < best_value
            else:
                is_best = current_metric > best_value
            
            # Save best model
            if is_best:
                best_model_path = save_checkpoint(
                    model, optimizer, epoch, aggr_metrics_val, config, 
                    'best_model.pth'
                )
                logger.info(f"New best model saved: {best_model_path}")
            
            # Early stopping check
            if early_stopping(current_metric, epoch):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best epoch was {early_stopping.best_epoch} with {config['key_metric']}: {early_stopping.best_score}")
                break

        # Save regular checkpoint
        if config.get('save_all', False):
            save_checkpoint(model, optimizer, epoch, aggr_metrics_train, config, f'model_epoch_{epoch}.pth')
        else:
            save_checkpoint(model, optimizer, epoch, aggr_metrics_train, config, 'model_last.pth')

        # Learning rate scheduling
        if hasattr(config, 'lr_step') and lr_step < len(config['lr_step']) and epoch == config['lr_step'][lr_step]:
            save_checkpoint(model, optimizer, epoch, aggr_metrics_train, config, f'model_epoch_{epoch}.pth')
            lr = lr * config['lr_factor'][lr_step]
            lr_step += 1
            logger.info(f'Learning rate updated to: {lr}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Save training history periodically
        if epoch % 10 == 0:
            history_path = os.path.join(config['output_dir'], 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)

        # Difficulty scheduling (if applicable)
        if config.get('harden', False) and check_progress(epoch):
            train_loader.dataset.update() 
            val_loader.dataset.update()

    # Save final training history
    history_path = os.path.join(config['output_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    # Save final model
    final_model_path = save_checkpoint(model, optimizer, epoch, aggr_metrics_val, config, 'final_model.pth')
    logger.info(f"Final model saved to: {final_model_path}")

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    # Final summary
    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    if best_model_path:
        logger.info('Best model saved at: {}'.format(best_model_path))
    if early_stopping.counter > 0:
        logger.info('Training stopped early after {} epochs without improvement'.format(early_stopping.counter))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    return best_value


if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
