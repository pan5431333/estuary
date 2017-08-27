name := "coursera-deeplearning-practice-in-scala"

version := "1.0"

scalaVersion := "2.12.0"


libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)

libraryDependencies += "org.springframework" % "spring-context" % "4.3.1.RELEASE"

libraryDependencies += "log4j" % "log4j" % "1.2.16"

//libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-core" % "2.1.7"
//
//libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-io-extra" % "2.1.7"
//
//libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-filters" % "2.1.7"
//
//resolvers += "stephenjudkins-bintray" at "http://dl.bintray.com/stephenjudkins/maven"
//
//libraryDependencies += "ps.tricerato" %% "pureimage" % "0.1.2"
//
//resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
//