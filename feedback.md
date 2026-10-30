  2. The model is overfitting heavily

  Training loss gets down to ~0.04, yet test accuracy is only 60%. The model is memorising the trainingimages. Two reasons:

  - Conv2d(3, 3, kernel_size=1) doesn't do what a CNN layer is meant to do. A 1×1 kernel only remixes the colour of each pixel on its own, so it can't detect edges or shapes. After it, fc1 connects 128·128·3=
    49,152 inputs to 128 units. That's about 6.3 million parameters looking at raw pixels, which is farmore than this dataset can support.
  - Dropout(0.01) has almost no effect. It turns off 1% of units. Values between 0.3 and 0.5 are typical.

  A small real CNN should do much better with far fewer parameters:

  self.features = nn.Sequential(
      nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64x64
      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x32
      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
  )
  self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(128, num_classes),
  )

  With this version you no longer need the input_size argument (or you can swap in nn.AdaptiveAvgPool2d(1) so any image size works).

  3. Data augmentation (training set only)

  Hand photos vary in angle, lighting and position. Random transforms teach the model to cope with thatvariation:

  train_tf = v2.Compose([
      v2.ToImage(), v2.Resize((128, 128)),
      v2.RandomHorizontalFlip(), v2.RandomRotation(15),
      v2.ColorJitter(brightness=0.2, contrast=0.2),
      v2.ToDtype(torch.float32, scale=True),
  ])

  Keep your current to_tensor for the test set.

  4. Smaller things

  - Call model.train() at the start of train(). evaluate() switches the model to eval mode, so if you ever train again afterwards, dropout would stay off without you noticing.
  - Remove the try/except blocks. In train.py, except Exception as e: raise e doesn't change anything. In evaluation.py it's worse: the function returns an error string, and the next function receives that
    string as if it were predictions. You'll get a confusing error later on instead of a clear traceback at the actual problem.
  - Track a validation split. Hold out about 10–20% of the training set with torch.utils.data.random_split, and print validation accuracy each epoch. That shows you exactly when overfitting starts, so you can
    decide how many epochs to train without looking at the test set.
  - Fix the parameter names in train(). Uppercase names like LEARNING_RATE are the Python convention for constants. Function parameters should be lowercase (lr, num_epochs).
  - Fix the outdated comments. # Load data with a 12/87 test/training split doesn't describe what the code does (the split comes from your folders). The RPSClassifier docstring also no longer mentions the conv
    layer.

  I'd start with #1, because you can't really judge your model until the evaluation is accurate. Then do #2, which should give the biggest accuracy jump. Want me to make any of these changes?