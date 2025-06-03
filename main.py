def main():
    print("Hello from mmpose-trainer!")

    # test gpu

    import torch

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for training.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU for training.")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    




if __name__ == "__main__":
    main()
