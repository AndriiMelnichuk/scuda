name := "scuda"

version := "0.1"

scalaVersion := "3.3.1"

// jcuda
libraryDependencies += "org.jcuda" % "jcuda" % "12.6.0"
libraryDependencies += "org.jcuda" % "jcuda-natives" % "12.6.0"

// jcublas
libraryDependencies += "org.jcuda" % "jcublas" % "12.6.0"
libraryDependencies += "org.jcuda" % "jcublas-natives" % "12.6.0"


// scalatest
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.17" % Test

// parallel collections
libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"