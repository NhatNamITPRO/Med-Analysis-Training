import argparse
import os
from train import Segformer3DBraTSTraining,Unet3DBraTSTraining,SegformerISICTraining,Unet2DISICTraining
from eval import Segformer3DBraTSEvaluating,Unet3DBraTSEvaluating,SegformerISICEvaluating,Unet2DISICEvaluating
from test import Segformer3DBraTSTesting,Unet3DBraTSTesting,SegformerISICTesting,Unet2DISICTesting
def main():
    parser = argparse.ArgumentParser(description="Train Segformer3D for BraTS2021 Dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name: Unet, Unet3D, Segformer, Segformer3D")
    parser.add_argument("--mode", type=str, required=True, help="Mode: traning, evaluating, testing")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--input_path", type=str, help="Path to the input")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to a model checkpoint for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--D", type=int, default=128, help="Depth of the 3D input")
    parser.add_argument("--H", type=int, default=128, help="Height of the 3D input")
    parser.add_argument("--W", type=int, default=128, help="Width of the 3D input")
    args = parser.parse_args()
    if args.mode in ["evaluating", "testing"] and args.model_checkpoint is None:
        parser.error(f"--model_checkpoint is required for {args.mode}")
    if args.mode in ["evaluating", "training"] and args.dataset_path is None:
        parser.error(f"--dataset_path is required for {args.mode}")
    if args.mode == "testing" and args.input_path is None:
        parser.error(f"--input_path is required for {args.mode}")
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    # Xử lý logic dựa trên chế độ (`mode`)
    if args.mode == "training":
        # Gọi class `Segformer3DBraTSTraining` với các tham số từ `args`
        if args.model_name == "Segformer3D":
            Segformer3DBraTSTraining.train(
                DATA_ROOT_PATH=args.dataset_path,
                CHECKPOINT_PATH=args.model_checkpoint,
                OUTPUT_PATH=args.output_dir,
                NUM_EPOCHS=args.epochs,
                BATCH_SIZE=args.batch_size,
                LR=args.learning_rate,
                D=args.D,
                H=args.H,
                W=args.W,
            )
        elif args.model_name == "Unet3D":
            Unet3DBraTSTraining.train(
                DATA_ROOT_PATH=args.dataset_path,
                CHECKPOINT_PATH=args.model_checkpoint,
                OUTPUT_PATH=args.output_dir,
                NUM_EPOCHS=args.epochs, 
                BATCH_SIZE=args.batch_size,
                LR=args.learning_rate,
                D=args.D,
                H=args.H,
                W=args.W,
            )
        elif args.model_name == "Segformer":
            SegformerISICTraining.train(
                dataset_path=args.dataset_path,
                checkpoint_path=args.model_checkpoint,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs, 
                height=args.H,
                width=args.W,
                lr=args.learning_rate,
            )
        elif args.model_name == "Unet2D":
            Unet2DISICTraining.train(
                dataset_path=args.dataset_path,
                checkpoint_path=args.model_checkpoint,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs, 
                height=args.H,
                width=args.W,
                lr=args.learning_rate,
            )

    elif args.mode == "evaluating":
        if args.model_name == "Segformer3D":
            Segformer3DBraTSEvaluating.eval(
                DATA_ROOT_PATH=args.dataset_path,
                CHECKPOINT_PATH=args.model_checkpoint,
                BATCH_SIZE=args.batch_size,
                D=args.D,
                H=args.H,
                W=args.W,
            )
        elif args.model_name == "Unet3D":
            Unet3DBraTSEvaluating.eval(
                DATA_ROOT_PATH=args.dataset_path,
                CHECKPOINT_PATH=args.model_checkpoint,
                BATCH_SIZE=args.batch_size,
                D=args.D,
                H=args.H,
                W=args.W,
            )
        elif args.model_name == "Segformer":
            SegformerISICEvaluating.eval(
                dataset_path=args.dataset_path,
                checkpoint_path=args.model_checkpoint,
                batch_size=args.batch_size,
                height=args.H,
                width=args.W,
            )
        elif args.model_name == "Unet2D":
            Unet2DISICEvaluating.eval(
                dataset_path=args.dataset_path,
                checkpoint_path=args.model_checkpoint,
                batch_size=args.batch_size,
                height=args.H,
                width=args.W,
            )
    elif args.mode == "testing":
        if args.model_name == "Segformer3D":
            Segformer3DBraTSTesting.predict(
                folder_path=args.input_path,
                checkpoint_path=args.model_checkpoint,
                output_dir=args.output_dir
            )
        elif args.model_name == "Unet3D":
            Unet3DBraTSTesting.predict(
                    folder_path=args.input_path,
                    checkpoint_path=args.model_checkpoint,
                    output_dir=args.output_dir
                )
        elif args.model_name == "Segformer":
            SegformerISICTesting.predict(
                    image_path=args.input_path,
                    checkpoint_path=args.model_checkpoint,
                    output_dir=args.output_dir
                )
        elif args.model_name == "Unet2D":
            Unet2DISICTesting.predict(
                    image_path=args.input_path,
                    checkpoint_path=args.model_checkpoint,
                    output_dir=args.output_dir,
                    height=args.H,
                    width=args.W,
                )


if __name__ == "__main__":
    main()