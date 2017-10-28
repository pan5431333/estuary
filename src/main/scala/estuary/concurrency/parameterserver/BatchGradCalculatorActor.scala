package estuary.concurrency.parameterserver

import akka.actor.{Actor, ActorLogging, ActorRef}
import breeze.linalg.DenseMatrix
import estuary.components.optimizer.ParallelOptimizer
import estuary.concurrency.MyMessage
import estuary.concurrency.parameterserver.BatchGradCalculatorActor.{StartTrain, TrainingDone}
import estuary.concurrency.parameterserver.ParameterServerActor._
import estuary.data.Reader
import estuary.model.Model

/**
  *
  * @tparam M Model Parameter
  * @tparam O Optimizer Parameter
  */
class BatchGradCalculatorActor[M <: AnyRef, O <: AnyRef](filePath: String,
                                                         dataReader: Reader,
                                                         model: Model[M],
                                                         parameterServer: ActorRef,
                                                         iteration: Int,
                                                         shuffleFunc: (DenseMatrix[Double], DenseMatrix[Double]) => Iterator[(DenseMatrix[Double], DenseMatrix[Double])],
                                                         updateFunc: (O, M, Int) => O,
                                                         modelToOp: M => O,
                                                         opToModel: O => M)
  extends Actor with ActorLogging {

  private[this] val (feature, label) = dataReader.read(filePath)
  private[this] var shuffledData: Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = shuffleFunc(feature, label)
  private[this] var miniBatchIndex: Int = 0
  private[this] var iterTime: Int = 0
  private[this] var currentFeature: DenseMatrix[Double] = _
  private[this] var currentLabel: DenseMatrix[Double] = _
  private[this] var grads: M = _
  private[this] lazy val miniBatchSize: Int = currentFeature.rows
  private[this] var manager: ActorRef = _

  override def receive: Actor.Receive = {
    case StartTrain =>
      manager = sender
      val init = model.init(feature.cols, label.cols)
      val initOp = modelToOp(init)
      parameterServer ! UpdateParams(initOp)
      parameterServer ! GetCurrentParams

    case CurrentParams(params) =>
      iter(params.asInstanceOf[O])
      if (iterTime < iteration) {
        parameterServer ! GetCurrentParamsForUpdate
      } else manager ! TrainingDone


    case CurrentParamsForUpdate(params, miniBatchTime) =>
      parameterServer ! UpdateParams(updateFunc(params.asInstanceOf[O], grads, miniBatchIndex))
      parameterServer ! GetCurrentParams
  }

  private def iter(params: O): Unit = {
    val cost = calculateCost(opToModel(params))
    val printCostUnit = math.max(currentFeature.rows / this.miniBatchSize / 5, 10)
    ParallelOptimizer.printCostInfo(cost, iterTime, miniBatchIndex, printCostUnit, log)
    if (miniBatchIndex % printCostUnit == 0) manager ! CostHistory(cost)
    grads = calculateGrads(opToModel(params))
  }

  private def calculateCost(params: M): Double = {
    if (shuffledData.hasNext) {
      val (onefeature, onelabel) = shuffledData.next()
      currentFeature = onefeature
      currentLabel = onelabel
      miniBatchIndex += 1
    } else {
      shuffledData = shuffleFunc(feature, label)
      miniBatchIndex = 0
      iterTime += 1
      val (onefeature, onelabel) = shuffledData.next()
      currentFeature = onefeature
      currentLabel = onelabel
    }
    model.forward(currentFeature, currentLabel, params)
  }

  private def calculateGrads(params: M): M = {
    model.backward(currentLabel, params)
  }
}

object BatchGradCalculatorActor {

  sealed trait BatchGradCalculatorActorMsg extends Serializable with MyMessage

  final case object StartTrain extends BatchGradCalculatorActorMsg

  final case object TrainingDone extends BatchGradCalculatorActorMsg

  case class WorkerConfig(address: String, filePath: String, dataReader: String)

}


