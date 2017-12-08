package estuary.components.support

import estuary.components.initializer.WeightsInitializer
import estuary.components.layers._
import estuary.model.Model

trait CanAutoInit[-For] {
  def init(foor: For, initializer: WeightsInitializer): Unit
}

object CanAutoInit {
//  implicit val anyLayerCanAutoInit: CanAutoInit[Layer] =
//    (foor, initializer) => {
//      foor match {
//        case f: ClassicLayer => implicitly[CanAutoInit[ClassicLayer]].init(f, initializer)
//        case f: ConvLayer => implicitly[CanAutoInit[ConvLayer]].init(f, initializer)
//        case f: DropoutLayer => implicitly[CanAutoInit[DropoutLayer]].init(f, initializer)
//        case f: PoolingLayer => implicitly[CanAutoInit[PoolingLayer]].init(f, initializer)
//        case _ => throw new Exception(s"Unsupported layer of type ${foor.getClass}")
//      }
//    }
//
//  implicit val anyModelCanAutoInit: CanAutoInit[Any] =
//    (foor, initializer) => {
//      foor match {
//        case f: Model => implicitly[CanAutoInit[Model]].init(f, initializer)
//        case f: Layer => implicitly[CanAutoInit[Layer]].init(f, initializer)
//        case _ => throw new Exception(s"Unsupported model of type ${foor.getClass}")
//      }
//    }
}
