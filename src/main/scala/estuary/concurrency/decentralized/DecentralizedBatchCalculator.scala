package estuary.concurrency.decentralized

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import breeze.linalg.DenseMatrix
import estuary.components.optimizer.ParallelOptimizer
import estuary.concurrency.MyMessage
import estuary.concurrency.decentralized.DecentralizedBatchCalculator._
import estuary.concurrency.decentralized.Manager.{CostHistory, StartTrain}
import estuary.data.Reader
import estuary.model.Model

import scala.collection.mutable

/**
  * Created by mengpan on 2017/10/28.
  */
class DecentralizedBatchCalculator[OptParam, ModelParam <: Any](id: Long,
                                                                filePath: String,
                                                                dataReader: Reader,
                                                                model: Model,
                                                                iteration: Int,
                                                                miniBatchFunc: (DenseMatrix[Double], DenseMatrix[Double]) => Iterator[(DenseMatrix[Double], DenseMatrix[Double])],
                                                                updateFunc: (OptParam, ModelParam, Int) => OptParam,
                                                                modelToOpFunc: ModelParam => OptParam,
                                                                opToModelFunc: OptParam => ModelParam,
                                                                avgOpFunc: Seq[OptParam] => OptParam)
  extends Actor with ActorLogging {

  private[this] var neibours: Seq[ActorRef] = _
  private[this] val neiboursParams: mutable.HashMap[ActorRef, OptParam] = new mutable.HashMap[ActorRef, OptParam]()
  private[this] val (feature, label) = dataReader.read(filePath)
  private[this] var myParams: OptParam = modelToOpFunc({model.init(feature.cols, label.cols); model.getParams[Seq[DenseMatrix[Double]]].asInstanceOf[ModelParam]})
  private[this] var shuffledData: Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = miniBatchFunc(feature, label)
  private[this] var miniBatchIndex: Int = 0
  private[this] var iterTime: Int = 0
  private[this] var currentFeature: DenseMatrix[Double] = _
  private[this] var currentLabel: DenseMatrix[Double] = _
  private[this] lazy val miniBatchSize: Int = currentFeature.rows
  private[this] var manager: ActorRef = _

  override def receive: Receive = {
    case Neibours(neibours) => this.neibours = neibours

    case CurrentParam(param) =>
      neiboursParams.+=(sender -> param.asInstanceOf[OptParam])
      if (neiboursParams.size == neibours.length) {
        myParams = avgOpFunc(myParams :: neiboursParams.values.toList)
        neiboursParams.clear()
        val grads = iter()
        myParams = updateFunc(myParams, grads, miniBatchIndex)
        if (iterTime < iteration)
          askNeiboursForParams(neibours)
        else {
          log.info(s"trained parameters on actor $self is: ${opToModelFunc(myParams)}")
          manager ! TrainingDone
        }
      }

    case GetParam =>
      sender ! CurrentParam(myParams)

    case StartTrain =>
      manager = sender
      askNeiboursForParams(neibours)

  }

  private def askNeiboursForParams(neibours: Seq[ActorRef]): Unit = {
    for (neibour <- neibours) {
      neibour ! GetParam
    }
  }

  private def iter(): ModelParam = {
    val cost = calculateCost(opToModelFunc(myParams))
    val printCostUnit = math.max(currentFeature.rows / this.miniBatchSize / 5, 10)
    ParallelOptimizer.printCostInfo(cost, iterTime, miniBatchIndex, printCostUnit, log)
    if (miniBatchIndex % printCostUnit == 0) manager ! CostHistory(cost)
    calculateGrads(opToModelFunc(myParams))
  }

  private def calculateCost(params: ModelParam): Double = {
    if (shuffledData.hasNext) {
      val (onefeature, onelabel) = shuffledData.next()
      currentFeature = onefeature
      currentLabel = onelabel
      miniBatchIndex += 1
    } else {
      shuffledData = miniBatchFunc(feature, label)
      miniBatchIndex = 0
      iterTime += 1
      val (onefeature, onelabel) = shuffledData.next()
      currentFeature = onefeature
      currentLabel = onelabel
    }
    model.forwardAndCalCost(currentFeature, currentLabel, params)
  }

  private def calculateGrads(params: ModelParam): ModelParam = {
    model.backwardWithGivenParams(currentLabel, params)
  }
}

object DecentralizedBatchCalculator {

  def props[OptParam, ModelParam](id: Long,
                                  filePath: String,
                                  dataReader: Reader,
                                  model: Model,
                                  iteration: Int,
                                  miniBatchFunc: (DenseMatrix[Double], DenseMatrix[Double]) => Iterator[(DenseMatrix[Double], DenseMatrix[Double])],
                                  updateFunc: (OptParam, ModelParam, Int) => OptParam,
                                  modelToOpFunc: ModelParam => OptParam,
                                  opToModelFunc: OptParam => ModelParam,
                                  avgOpFunc: Seq[OptParam] => OptParam): Props = {
    Props(classOf[DecentralizedBatchCalculator[OptParam, ModelParam]], id, filePath, dataReader, model, iteration, miniBatchFunc, updateFunc, modelToOpFunc, opToModelFunc, avgOpFunc)
  }

  sealed trait DecentralizedBatchCalculatorMsg extends MyMessage

  final case class Neibours(actors: Seq[ActorRef]) extends DecentralizedBatchCalculatorMsg

  final case object GetParam extends DecentralizedBatchCalculatorMsg

  final case class CurrentParam(param: Any) extends DecentralizedBatchCalculatorMsg

  final case object TrainingDone extends DecentralizedBatchCalculatorMsg

}
