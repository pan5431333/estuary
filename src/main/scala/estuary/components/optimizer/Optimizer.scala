package estuary.components.optimizer

import java.text.SimpleDateFormat
import java.util.Calendar

import breeze.linalg.DenseMatrix
import estuary.components.Exception.GradientExplosionException
import estuary.model.Model
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer


/**
  * Optimizer Interface, all of whose implementations MUST implement abstract method: "optimize"
  *
  * @note This optimizer can be ONLY used for optimizing Machine Learning-like
  *       problems, which means that input-output (i.e. feature-label) data are needed.
  *       NOT for general optimization or mathematical planning problem.
  */
trait Optimizer {
  protected val logger: Logger

  /**
    * Optimizing Machine Learning-like models' parameters on a training data set (feature, label).
    *
    * @param feature      feature matrix
    * @param label        label matrix in one-hot representation
    * @param initParams   Initialized parameters.
    * @param forwardFunc  The cost function.
    *                     inputs: (feature, label, params) of type (DenseMatrix[Double], DenseMatrix[Double], T)
    *                     output: cost of type Double.
    * @param backwardFunc A function calculating gradients of all parameters.
    *                     input: (label, params) of type (DenseMatrix[Double], T)
    *                     output: gradients of params of type T.
    * @return Trained parameters.
    */
  def optimize(feature: DenseMatrix[Double], label: DenseMatrix[Double])
              (initParams: Seq[DenseMatrix[Double]])
              (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Double)
              (backwardFunc: (DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]]

  protected def handleGradientExplosionException(params: Any, paramSavePath: String): Unit

  protected var iteration: Int = _
  protected var learningRate: Double = _
  protected var paramSavePath: String = System.getProperty("user.dir")
  protected var exceptionCount: Int = 0
  protected var minCost: Double = 0.0

  def setIteration(iteration: Int): this.type = {
    this.iteration = iteration
    this
  }

  def setLearningRate(learningRate: Double): this.type = {
    assert(learningRate > 0, "Nonpositive learning rate: " + learningRate)
    this.learningRate = learningRate
    this
  }

  def setParamSavePath(path: String): this.type = {
    this.paramSavePath = paramSavePath
    this
  }

  /** Storing cost history after every iteration. */
  val costHistory: ArrayBuffer[Double] = new ArrayBuffer[Double]()

  @throws[GradientExplosionException]
  protected def checkGradientsExplosion(nowCost: Double, minCost: Double): Unit = {
    if (nowCost - minCost > 10)
      throw new GradientExplosionException(s"Cost is rising ($minCost -> $nowCost), it seems that gradient explosion happens...")
  }

  protected def saveDenseMatricesToDisk(params: Seq[DenseMatrix[_]], paramSavePath: String): Unit = {
    val modelParams = params
    val currentTime = Calendar.getInstance().getTime
    val timeFormat = new SimpleDateFormat("yyyyMMddHHmmss")
    val fileName = paramSavePath + "/" + timeFormat.format(currentTime) + ".txt"
    Model.saveDenseMatricesToDisk(modelParams, fileName)
    logger.warn(s"Something wrong happened during training, the current parameters have been save to $fileName")
  }
}
