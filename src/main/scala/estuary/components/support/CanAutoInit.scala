package estuary.components.support

import estuary.components.initializer.WeightsInitializer
import shapeless.{:+:, CNil, Coproduct, Generic, Inl, Inr, Lazy}

trait CanAutoInit[-For] {
  def init(foor: For, initializer: WeightsInitializer): Unit
}

object CanAutoInit {
  implicit val cnilCanAutoInit: CanAutoInit[CNil] = new CanAutoInit[CNil] {
    override def init(foor: CNil, initializer: WeightsInitializer): Unit =
      throw new UnsupportedClassVersionError("Invoke init() on CNil")
  }

  implicit def coproductCanAutoInit[H, T <: Coproduct](implicit hCanAutoInit: CanAutoInit[H],
                                                       tCanAutoInit: CanAutoInit[T]): CanAutoInit[H :+: T] = new CanAutoInit[H :+: T] {
    override def init(foor: :+:[H, T], initializer: WeightsInitializer): Unit = foor match {
      case Inl(h) => hCanAutoInit.init(h, initializer)
      case Inr(t) => tCanAutoInit.init(t, initializer)
    }
  }

  implicit def genericCanAutoInit[A, C <: Coproduct](implicit generic: Generic.Aux[A, C],
                                                     cCanAutoInit: Lazy[CanAutoInit[C]]): CanAutoInit[A] = new CanAutoInit[A] {
    override def init(foor: A, initializer: WeightsInitializer): Unit = cCanAutoInit.value.init(generic.to(foor), initializer)
  }
}
