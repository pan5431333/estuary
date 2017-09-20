# estuary
Estuary is a Deep Learning framework in Scala built from scratch. Data structures like DenseVector, DenseMatrix, SparseVector and SparseMatrix in Breeze library are used to perform matrix multiplication operation both on CPU and on GPU. 

A quick start is as follows: 

First, we need to setup a model: 
  val nnModel: Model = new NeuralNetworkModel()
    .setWeightsInitializer(HeInitializer) //can also be XavierInitializer, NormalInitializer, for ReLU HeInitializer is recommeded
    .setRegularizer(VoidRegularizer) //can also be L2Regularizer, L1Regularizer
    .setOptimizer(AdamOptimizer(miniBatchSize = 64)) //can also be GDOptimizer, SGDOptimizer 
    .setHiddenLayerStructure(
      ReluLayer(400, batchNorm = true), //also have SigmoidLayer, TanhLayer, SoftmaxLayer 
      ReluLayer(200, batchNorm = true)
    )
    .setOutputLayerStructure(SoftmaxLayer())
    .setLearningRate(0.001)
    .setIterationTime(10)
    
After model has been setup, we can call train() method in that model: 
  val trainedModel: Model = nnModel.train(trainingFeature, trainingLabel)

where trainingFeature is a DenseMatrix of shape (n, p), where n is the number of training examples, p is the input feature's dimension, 
and trainingLabel is a DenseVector of length n. 

Then we can call predict() method in the trained model to predict new features: 
  val yPredicted = trainedModel.predict(testFeature)
  
where testFeature is a DenseMatrix of shape (m, p) where m is the number of test features and p is the input feature's dimension. 

Finally we can call accuracy() method in the trained model to calculate the accuracy of our prediction: 
  val testAccuracy = trainedModel.accuracy(testLabel, yPredicted)
  
That's the quick start for estuary. Further detailed documentation will be coming soon. Thank you very much. 
