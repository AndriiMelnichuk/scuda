name := "scuda"

version := "0.1"

scalaVersion := "3.3.1"

// jcuda
libraryDependencies += "org.jcuda" % "jcuda" % "11.8.0"
libraryDependencies += "org.jcuda" % "jcuda-natives" % "11.8.0"

// scalatest
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.17" % Test
