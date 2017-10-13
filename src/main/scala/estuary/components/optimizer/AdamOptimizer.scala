package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import breeze.numerics.{pow, sqrt}
import estuary.components.optimizer.AdamOptimizer.AdamParam

/**
  * Adam Optimizer, a very efficient and recommended optimizer for Deep Neural Network.
  */
class AdamOptimizer extends Optimizer with MiniBatchable with Heuristic {
  protected var momentumRate: Double = 0.9
  protected var adamRate: Double = 0.999

  def setMomentumRate(momentum: Double): this.type = {
    this.momentumRate = momentum
    this
  }

  def setAdamRate(adamRate: Double): this.type = {
    this.adamRate = adamRate
    this
  }




  /**
    * Implementation of Optimizer.optimize(). Optimizing Machine Learning-like models'
    * parameters on a training dataset (feature, label).
    *
    * @param feature      DenseMatrix of shape (n, p) where n: the number of training
    *                     examples, p: the dimension of input feature.
    * @param label        DenseMatrix of shape (n, q) where n: the number of training examples,
    *                     q: number of distinct labels.
    * @param initParams   Initialized parameters.
    * @param forwardFunc  The cost function.
    *                     inputs: (feature, label, params) of type
    *                     (DenseMatrix[Double], DenseMatrix[Double], T)
    *                     output: cost of type Double.
    * @param backwardFunc A function calculating gradients of all parameters.
    *                     input: (label, params) of type (DenseMatrix[Double], T)
    *                     output: gradients of params of type T.
    * @return Trained parameters.
    */
  override def optimize(feature: DenseMatrix[Double], label: DenseMatrix[Double])
                       (initParams: Seq[DenseMatrix[Double]])
                       (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Double)
                       (backwardFunc: (DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    val initMomentum = getInitAdam(initParams)
    val initAdam = getInitAdam(initParams)
    val initAdamParam = AdamParam(initParams, initMomentum, initAdam)
    val numExamples = feature.rows
    val printMiniBatchUnit = numExamples / this.miniBatchSize / 5 //for each iteration, only print minibatch cost FIVE times.

    (0 to this.iteration).foldLeft[AdamParam](initAdamParam) { case (preParam, iterTime) =>
      val minibatches = getMiniBatches(feature, label)
      minibatches.zipWithIndex.foldLeft[AdamParam](preParam) { case (preBatchParams, ((batchFeature, batchLabel), miniBatchTime)) =>
        val cost = forwardFunc(batchFeature, batchLabel, preBatchParams.modelParam)
        val grads = backwardFunc(batchLabel, preBatchParams.modelParam)

        if (miniBatchTime % printMiniBatchUnit == 0)
          logger.info("Iteration: " + iterTime + "|=" + "=" * (miniBatchTime / printMiniBatchUnit) + ">> Cost: " + cost)
        costHistory.+=(cost)

        updateFunc(preBatchParams, grads, iterTime * miniBatchSize + miniBatchTime)
      }
    }.modelParam
  }

  /**
    * Initialize momentum or RMSProp parameters to all zeros.
    */
  protected def getInitAdam(modelParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    modelParams.par.map { m => DenseMatrix.zeros[Double](m.rows, m.cols) }.seq
  }

  /**
    * Update model parameters, momentum parameters and RMSProp parameters by Adam method.
    *
    * @param params        model parameters, momentum parameters and RMSProp parameters, where T is the type of model parameters.
    * @param grads         Gradients of model parameters on current iteration.
    * @param miniBatchTime Current minibatch time.
    * @return Updated model parameters, momentum parameters and RMSProp parameters.
    */
  def updateFunc(params: AdamParam, grads: Seq[DenseMatrix[Double]], miniBatchTime: Int): AdamParam = {
    val (ps, vs, ws) = params match {
      case AdamParam(a, b, c) => (a, b, c)
    }

    val updatedParams = ps.zip(vs).zip(ws).zip(grads).par.map { case (((p, v), a), grad) =>
      val vN = v * this.momentumRate + grad * (1.0 - this.momentumRate)
      val vNCorrected = vN / (1.0 - pow(this.momentumRate, miniBatchTime.toDouble + 1.0))

      val aN = a * this.adamRate + (1.0 - this.adamRate) * pow(grad, 2.0)
      val aNCorrected = aN / (1.0 - pow(this.adamRate, miniBatchTime.toDouble + 1.0))

      val pN = p - learningRate * vNCorrected /:/ (sqrt(aNCorrected) + 1E-8)
      (pN, vN, aN)
    }.seq.toList

    val (modelParams, momentumAndAdam) = updatedParams.unzip(f => (f._1, (f._2, f._3)))

    val (momentum, adam) = momentumAndAdam.unzip(f => (f._1, f._2))

    AdamParam(modelParams.asInstanceOf[Seq[DenseMatrix[Double]]], momentum.asInstanceOf[Seq[DenseMatrix[Double]]], adam.asInstanceOf[Seq[DenseMatrix[Double]]])
  }
}

object AdamOptimizer {

  case class AdamParam(modelParam: Seq[DenseMatrix[Double]], momentumParam: Seq[DenseMatrix[Double]], adamParam: Seq[DenseMatrix[Double]])

  def apply(miniBatchSize: Int = 64, momentumRate: Double = 0.9, adamRate: Double = 0.999): AdamOptimizer = {
    new AdamOptimizer()
      .setMiniBatchSize(miniBatchSize)
      .setAdamRate(adamRate)
      .setMomentumRate(momentumRate)
  }
}
