import org.scalatest.FunSuite

import scala.io.Source

/**
  * Created by mengpan on 2017/11/25.
  */
class CifarTenReaderTest extends FunSuite{
  test("Test for Cifar Ten Label Reading on Mac") {
    Source.fromFile("""/Users/mengpan/Downloads/CatData/trainLabels.csv""")
  }


}
