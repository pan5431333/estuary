package estuary.components.support

import breeze.linalg.DenseMatrix
import estuary.components.layers._
import shapeless.{:+:, CNil, Coproduct, Generic, Inl, Inr}

trait CanForward[-By, -Input, +Output] {
  def forward(input: Input, by: By): Output
}

object CanForward {
  implicit val cnilCanForward = new CanForward[CNil, DenseMatrix[Double], DenseMatrix[Double]] {
    override def forward(input: DenseMatrix[Double], by: CNil) = throw new UnsupportedOperationException("Invoke forward() on CNil")
  }

  implicit def coproductCanForward[L, R <: Coproduct](implicit lCanForward: CanForward[L, DenseMatrix[Double], DenseMatrix[Double]],
                                                      rCanForward: CanForward[R, DenseMatrix[Double], DenseMatrix[Double]]) =
    new CanForward[L :+: R, DenseMatrix[Double], DenseMatrix[Double]] {
      override def forward(input: DenseMatrix[Double], by: :+:[L, R]): DenseMatrix[Double] = {
        by match {
          case Inl(l) => lCanForward.forward(input, l)
          case Inr(r) => rCanForward.forward(input, r)
        }
      }
    }

  implicit def genericCanForward[A, C <: Coproduct](implicit generic: Generic.Aux[A, C],
                                                    cCanForward: CanForward[C, DenseMatrix[Double], DenseMatrix[Double]]) =
    new CanForward[A, DenseMatrix[Double], DenseMatrix[Double]] {
      override def forward(input: DenseMatrix[Double], by: A): DenseMatrix[Double] = cCanForward.forward(input, generic.to(by))
    }
}
