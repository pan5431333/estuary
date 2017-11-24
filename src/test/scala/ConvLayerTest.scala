import breeze.linalg.DenseMatrix
import estuary.components.initializer.OneInitializer
import org.scalatest.FunSuite
import estuary.components.layers.{ConvLayer, ReluConvLayer}
import estuary.components.layers.ConvLayer.{ConvSize, Filter}
import estuary.model.Model

class ConvLayerTest extends FunSuite {

  test("Test for Im2Col for 3*3 image and 2*2 filter") {
    val a = DenseMatrix.create[Double](1, 9, Array(1, 4, 7, 2, 5, 8, 3, 6, 9).map(_.toDouble))
    val b = ConvLayer.im2col(a, ConvSize(3, 3, 1), Filter(2, 0, 1, 1, 1))
    assert(b === DenseMatrix.create[Double](4, 4, Array(1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9).map(_.toDouble)))
  }

  test("Test for Im2Col for 2*2 image and 1*1 filter ") {
    val a = DenseMatrix.create[Double](1, 4, Array(1, 3, 2, 4).map(_.toDouble))
    val b = ConvLayer.im2col(a, ConvSize(2, 2, 1), Filter(1, 0, 1, 1, 1))
    assert(b === DenseMatrix.create[Double](4, 1, Array(1, 2, 3, 4).map(_.toDouble)))
  }

  test("Test for Im2Col for 2*2 image and 2*2 filter ") {
    val a = DenseMatrix.create[Double](1, 4, Array(1, 3, 2, 4).map(_.toDouble))
    val b = ConvLayer.im2col(a, ConvSize(2, 2, 1), Filter(2, 0, 1, 1, 1))
    assert(b === DenseMatrix.create[Double](1, 4, Array(1, 2, 3, 4).map(_.toDouble)))
  }

  test("Test for Im2Col for 2*2*2 image and 2*2*2 filter ") {
    val a = DenseMatrix.create[Double](1, 8, Array(1, 3, 2, 4, 5, 7, 6, 8).map(_.toDouble))
    val b = ConvLayer.im2col(a, ConvSize(2, 2, 2), Filter(2, 0, 1, 2, 1))
    assert(b === DenseMatrix.create[Double](1, 8, Array(1, 2, 3, 4, 5, 6, 7, 8).map(_.toDouble)))
  }

  test("Test for Im2Col for 4*4 image and 2*2 filter with stride 2") {
    val a = DenseMatrix.create[Double](1, 16, Array(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16).map(_.toDouble))
    val b = ConvLayer.im2col(a, ConvSize(4, 4, 1), Filter(2, 0, 2, 1, 1))
    assert(b === DenseMatrix.create[Double](4, 4, Array(1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16).map(_.toDouble)))
  }

  test("Test for Col2Im for 1 2*2 image") {
    val a = DenseMatrix.create[Double](4, 1, Array(12, 16, 24, 28).map(_.toDouble))
    val b = ConvLayer.col2im(a, ConvSize(2, 2, 1))
    println(b)
    assert(b === DenseMatrix.create[Double](1, 4, Array(12, 16, 24, 28).map(_.toDouble)))
  }

  test("Test for Col2Im for 3 2*2 image") {
    val a = DenseMatrix.create[Double](12, 1, Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).map(_.toDouble))
    val b = ConvLayer.col2im(a, ConvSize(2, 2, 1))
    println(b)
    assert(b === DenseMatrix.create[Double](3, 4, Array(1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12).map(_.toDouble)))
  }

  test("Test for Col2Im for 2 2*2*3 image") {
    val a = DenseMatrix.create[Double](8, 3, Array(1, 2, 3, 4, 13, 14, 15, 16, 5, 6, 7, 8, 17, 18, 19, 20, 9, 10, 11, 12, 21, 22, 23, 24).map(_.toDouble))
    val b = ConvLayer.col2im(a, ConvSize(2, 2, 3))
    println(b)
    assert(b === DenseMatrix.create[Double](2, 12, Array(1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24).map(_.toDouble)))
  }

  test("Test for convolutional layer forward") {
    val filter = Filter(2, 0, 1, 1, 3)
    filter.init(OneInitializer)
    val layer = ReluConvLayer(filter, ConvSize(3, 3, 1))
    val yP = DenseMatrix.create[Double](1, 9, Array(1, 2, 3, 4, 5, 6, 7, 8, 9).map(_.toDouble))
    val yC = layer.forward(yP)
    println(yC)
    assert(yC === DenseMatrix.create[Double](1, 12, Array(13, 25, 17, 29, 13, 25, 17, 29, 13, 25, 17, 29).map(_.toDouble)))
  }

  test("Test for imGrad2Col with 1 2*2*1 image") {
    val a = DenseMatrix.create[Double](1, 4, Array(1, 2, 3, 4).map(_.toDouble))
    val b = ConvLayer.imGrad2Col(a, ConvSize(2, 2, 1))
    println(b)
    assert(b === DenseMatrix.create[Double](4, 1, Array(1, 2, 3, 4).map(_.toDouble)))
  }

  test("Test for imGrad2Col with 2 2*2*1 image") {
    val a = DenseMatrix.create[Double](2, 4, Array(1, 5, 2, 6, 3, 7, 4, 8).map(_.toDouble))
    val b = ConvLayer.imGrad2Col(a, ConvSize(2, 2, 1))
    println(b)
    assert(b === DenseMatrix.create[Double](8, 1, Array(1, 2, 3, 4, 5, 6, 7, 8).map(_.toDouble)))
  }

  test("Test for imGrad2Col with 2 2*2*2 image") {
    val a = DenseMatrix.create[Double](2, 8, Array(1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16).map(_.toDouble))
    val b = ConvLayer.imGrad2Col(a, ConvSize(2, 2, 2))
    println(b)
    assert(b === DenseMatrix.create[Double](8, 2, Array(1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16).map(_.toDouble)))
  }

  test("Test for convolutional layer backward") {
    val filter = Filter(2, 0, 1, 1, 3)
    filter.init(OneInitializer)
    val layer = ReluConvLayer(filter, ConvSize(3, 3, 1))

    val yP = DenseMatrix.create[Double](1, 9, Array(1, 2, 3, 4, 5, 6, 7, 8, 9).map(_.toDouble))
    val yC = layer.forward(yP)

    assert(yC === DenseMatrix.create[Double](1, 12, Array(13.0, 25.0, 17.0, 29.0, 13.0, 25.0, 17.0, 29.0, 13.0, 25.0, 17.0, 29.0)))

    val dY = DenseMatrix.create[Double](1, 12, Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).map(_.toDouble))

    val (dYP, dF) = layer.backward(dY, None)

//    assert(dYP === DenseMatrix.create[Double](4, 4, Array(13.0, 25.0, 17.0, 29.0, 13.0, 25.0, 17.0, 29.0, 13.0, 25.0, 17.0, 29.0)))
    assert(dF === DenseMatrix.create[Double](5, 3, Array(35,65,45,75,10,83,161,109,187,26,131,257,173,299,42).map(_.toDouble)))
  }

}
