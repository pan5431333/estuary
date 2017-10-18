package estuary.concurrency

import akka.actor.{Actor, ActorLogging}
import estuary.concurrency.ParameterServer._

class ParameterServer[B](initParams: B) extends Actor with ActorLogging{
  private[this] var parameters: B = initParams
  private[this] var miniBatchTime: Int = 0

  override def receive: Actor.Receive = {
    case GetCurrentParams => sender ! CurrentParams(parameters)

    case UpdateParams(newParams: B) =>
      parameters = newParams
      miniBatchTime += 1

    case GetCurrentParamsForUpdate => sender ! CurrentParamsForUpdate(parameters, miniBatchTime)
  }
}

object ParameterServer {
  sealed trait ParameterServerMsg
  case object GetCurrentParams extends ParameterServerMsg
  case object GetCurrentParamsForUpdate extends ParameterServerMsg
  case class UpdateParams[B](newParams: B) extends ParameterServerMsg
  case class CurrentParams[B](params: B) extends ParameterServerMsg
  case class CurrentParamsForUpdate[B](params: B, miniBatchTime: Int) extends ParameterServerMsg

}
