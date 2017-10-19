package estuary.concurrency

import akka.actor.{Actor, ActorLogging}
import estuary.concurrency.ParameterServerActor._

class ParameterServerActor[B](initParams: B) extends Actor with ActorLogging{
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

object ParameterServerActor {
  sealed trait ParameterServerActorMsg
  case object GetCurrentParams extends ParameterServerActorMsg
  case object GetCurrentParamsForUpdate extends ParameterServerActorMsg
  case class UpdateParams[B](newParams: B) extends ParameterServerActorMsg
  case class CurrentParams[B](params: B) extends ParameterServerActorMsg
  case class CurrentParamsForUpdate[B](params: B, miniBatchTime: Int) extends ParameterServerActorMsg

}
