package estuary.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{pow, sqrt}

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

  case class AdamParam[T <: Seq[DenseMatrix[Double]]](modelParam: T, momentumParam: T, adamParam: T)

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
    * @tparam T The type of model parameters
    * @return Trained parameters.
    */
  override def optimize[T <: DenseMatrix[Double]](feature: DenseMatrix[Double], label: DenseMatrix[Double])
                                                      (initParams: Seq[T])
                                                      (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], Seq[T]) => Double)
                                                      (backwardFunc: (DenseMatrix[Double], Seq[T]) => Seq[T]): Seq[T] = {
    val initMomentum = getInitAdam(initParams)
    val initAdam = getInitAdam(initParams)
    val initAdamParam = AdamParam[Seq[T]](initParams, initMomentum, initAdam)
    val numExamples = feature.rows
    val printMiniBatchUnit = numExamples / this.miniBatchSize / 5 //for each iteration, only print minibatch cost FIVE times.

    (0 to this.iteration).foldLeft[AdamParam[Seq[T]]](initAdamParam) { case (preParam, iterTime) =>
      val minibatches = getMiniBatches(feature, label)
      minibatches.zipWithIndex.foldLeft[AdamParam[Seq[T]]](preParam) { case (preBatchParams, ((batchFeature, batchLabel), miniBatchTime)) =>
        val cost = forwardFunc(batchFeature, batchLabel, preBatchParams.modelParam)
        val grads = backwardFunc(batchLabel, preBatchParams.modelParam)

        if (miniBatchTime % printMiniBatchUnit == 0)
          logger.info("Iteration: " + iterTime + "|=" + "=" * (miniBatchTime / 10) + ">> Cost: " + cost)
        costHistory.+=(cost)

        updateFunc(preBatchParams, grads, iterTime * miniBatchSize + miniBatchTime)
      }
    }.modelParam
  }

  /**
    * Initialize momentum or RMSProp parameters to all zeros.
    *
    * @param modelParams
    * @tparam T
    * @return
    */
  private def getInitAdam[T <: DenseMatrix[Double]](modelParams: Seq[T]): Seq[T] = {
    modelParams.par.map { m => DenseMatrix.zeros[Double](m.rows, m.cols) }.seq.asInstanceOf[Seq[T]]
  }

  /**
    * Update model parameters, momentum parameters and RMSProp parameters by Adam method.
    *
    * @param params        model parameters, momentum parameters and RMSProp parameters of type
    *                      AdamParam[T], where T is the type of model parameters.
    * @param grads         Gradients of model parameters on current iteration.
    * @param miniBatchTime Current minibatch time.
    * @tparam T the type of model paramaters
    * @return Updated model parameters, momentum parameters and RMSProp parameters.
    */
  private def updateFunc[T <: Seq[DenseMatrix[Double]]](params: AdamParam[T], grads: T, miniBatchTime: Int): AdamParam[T] = {
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

    AdamParam[T](modelParams.asInstanceOf[T], momentum.asInstanceOf[T], adam.asInstanceOf[T])
  }
}

object AdamOptimizer {
  def apply(miniBatchSize: Int = 64, momentumRate: Double = 0.9, adamRate: Double = 0.999): AdamOptimizer = {
    new AdamOptimizer()
      .setMiniBatchSize(miniBatchSize)
      .setAdamRate(adamRate)
      .setMomentumRate(momentumRate)
  }
}
