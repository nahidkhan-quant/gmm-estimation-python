
load("C:/Users/Raed/Desktop/NIU/classes/second2024/maria- methods in economics 692/homwork/ps5/PS5v3.RData")
View(GMMdata)

library(gmm) # function"gmm" is in that package

GMmat<-data.matrix(GMMdata)# Let's convert our data frame into a matrix:


# (b)Estimate β0 and β1
# g(theta,x) is the same thing as our m(W,theta)
# Here theta=(beta,gamma)'
moments<-function(theta, x){
  g <- x[,'y']*exp(-theta[1]-theta[2]*x[,'x'])-exp(0.5)
  g1 <- (x[,'y']*exp(-theta[1]-theta[2]*x[,'x'])-exp(0.5))*x[,'z1']
  return(cbind(g,g1))
} 

t0=c(0.3,5)# Initial values
GMM1<-gmm(moments,GMmat,t0=c(0.1,1),type="twoStep")# Two-Step Gmm
summary(GMM1) # View the values of J-test



# (c)#come up with an additional unconditional moment restriction that would allow you to OVER identify β0 and β1


moment2=function(theta, x){
  g <- x[,'y']*exp(-theta[1]-theta[2]*x[,'x'])-exp(0.5)
  g1 <- (x[,'y']*exp(-theta[1]-theta[2]*x[,'x'])-exp(0.5))*x[,'z1']
  g2 <- (x[,'y']*exp(-theta[1]-theta[2]*x[,'x'])-exp(0.5))*(x[,'z1'])*(x[,'z2'])
  return(cbind(g,g1,g2))
}


# (d)

t0=c(0.3,5)# Initial values

GMM2<-gmm(moment2,GMmat,t0=c(0.1,1),type="twoStep")

summary(GMM2) # View the values of J-test

