package estuary.components.support

import estuary.components.initializer.WeightsInitializer
import estuary.components.layers._

trait CanAutoInit[For] {
  def init(foor: For, initializer: WeightsInitializer): Unit
}

object CanAutoInit {
//  implicit val cnilCanAutoInit: CanAutoInit[CNil] = new CanAutoInit[CNil] {
//    override def init(foor: CNil, initializer: WeightsInitializer): Unit =
//      throw new UnsupportedClassVersionError("Invoke init() on CNil")
//  }
//
//  implicit def coproductCanAutoInit[H, T <: Coproduct](implicit hCanAutoInit: CanAutoInit[H],
//                                                       tCanAutoInit: CanAutoInit[T]): CanAutoInit[H :+: T] = new CanAutoInit[H :+: T] {
//    override def init(foor: :+:[H, T], initializer: WeightsInitializer): Unit = foor match {
//      case Inl(h) => hCanAutoInit.init(h, initializer)
//      case Inr(t) => tCanAutoInit.init(t, initializer)
//    }
//  }
//
//  implicit def genericCanAutoInit[A, C <: Coproduct](implicit generic: Generic.Aux[A, C],
//                                                     cCanAutoInit: Lazy[CanAutoInit[C]]): CanAutoInit[A] = new CanAutoInit[A] {
//    override def init(foor: A, initializer: WeightsInitializer): Unit = cCanAutoInit.value.init(generic.to(foor), initializer)
//  }

  implicit def canAutoInitLayer = new CanAutoInit[Layer] {
    override def init(foor: Layer, initializer: WeightsInitializer): Unit = foor match {
      case c: ClassicLayer => c.init(initializer)
      case con: ConvLayer => con.init(initializer)
      case d: DropoutLayer => d.init(initializer)
      case p: PoolingLayer => p.init(initializer)
      case _ => throw new UnsupportedOperationException(s"Initialization on $foor is unsupported")
    }
  }
}
