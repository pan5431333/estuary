package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import estuary.model.Model
import org.slf4j.{Logger, LoggerFactory}

class SGDParallelOptimizer(override val iteration: Int,
                           override val learningRate: Double,
                           override val paramSavePath: String,
                           override val miniBatchSize: Int,
                           val nTasks: Int)
  extends SGDOptimizer(iteration, learningRate, paramSavePath, miniBatchSize)
    with AbstractParallelOptimizer[Seq[DenseMatrix[Double]], Seq[DenseMatrix[Double]]] {

  override val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model[Seq[DenseMatrix[Double]]], initParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    updateParameterServer(initParams)

    val parBatches = genParBatches(feature, label)
    val modelInstances = parBatches.map(_ => model.copyStructure)

    for ((batch, model) <- parBatches.zip(modelInstances)) {
      for (i <- (0 until iteration).toIterator) {
        for (((feature, label), miniBatchTime) <- getMiniBatches(batch._1, batch._2).zipWithIndex) {
          val printMiniBatchUnit = math.max(feature.rows / this.miniBatchSize / 5, 10)
          val params = fetchParameterServer()
          val cost = model.forward(feature, label, params)

          if (miniBatchTime % printMiniBatchUnit == 0) {
            ParallelOptimizer.printCostInfo(cost, i, miniBatchTime, printMiniBatchUnit, logger)
            addCostHistory(cost)
          }

          val grads = model.backward(label, params)
          updateParameterServer(grads, miniBatchTime)
        }
      }
    }

    fetchParameterServer()
  }

  protected def updateParameterServer(grads: Seq[DenseMatrix[Double]], miniBatchTime: Int): Unit = {
    val nowParams = fetchParameterServer()
    val newParams = nowParams.zip(grads).par.map { case (param, grad) =>
      updateFunc(param, grad, miniBatchTime)
    }.seq
    updateParameterServer(newParams)
  }

  protected def updateFunc(params: DenseMatrix[Double], grads: DenseMatrix[Double], miniBatchTime: Int): DenseMatrix[Double] =
    updateFunc(List(params), List(grads), miniBatchTime).head
}

object SGDParallelOptimizer {
  def apply(iteration: Int = 100, learningRate: Double = 0.001, paramSavePath: String = System.getProperty("user.dir"), miniBatchSize: Int = 64, nTasks: Int = 4): SGDParallelOptimizer = {
    new SGDParallelOptimizer(iteration, learningRate, paramSavePath, miniBatchSize, nTasks)
  }
}


