#version 330 core
layout (location = 0) in vec4 inPos;

out vec4 pos;
out vec4 literalPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform float time;
uniform samplerBuffer tex;

#define PI2 (3.1415*2.0)
#define NUM_WAVES 3

struct wave
{
  vec2 vec; // wavevector
  float amp;
  float freq;
  float phase;
};

void setWaveVec(out wave wav, in float waveLen, in vec2 dir)
{
  wav.vec = normalize(dir)*PI2/waveLen;
}

vec3 gerstner(in vec2 x, float t, in wave wav) // obviously unoptimized
{
  vec2 newX = x - normalize(wav.vec) * wav.amp * sin(dot(wav.vec,x) - t*wav.freq + wav.phase);
  float newY = wav.amp*cos(dot(wav.vec, x) - t*wav.freq);
  return vec3(newX.x, newY, newX.y);
}

vec3 gerstner(in vec2 x, float t, in wave waves[NUM_WAVES])
{
  vec2 sumX = x;
  float sumY = 0.0;

  for(int i = 0; i < NUM_WAVES; i++)
  {
    float param = dot(waves[i].vec,x) - t*waves[i].freq + waves[i].phase;
    sumX = sumX - normalize(waves[i].vec) * waves[i].amp * sin(param);
    sumY = sumY + waves[i].amp*cos(param);
  }

  return vec3(sumX.x, sumY, sumX.y);
}

float hypTan(float x)
{
  if (x >= 4.0)
  {
    return 1.0;
  }
  float y = exp(2.*x);
  return max((y-1.)/(y+1.), 0.2);
//  float z = x*x;
//  return (x)/(1.+z*z/(3.+z/(.5+z)));
}



vec3 mutateVec(in vec3 x)
{
  // vec2 vec = x.xz*2.0;
  //
  // wave waves[NUM_WAVES];
  // setWaveVec(waves[0], 12., vec2(25,45));
  // waves[0].amp = 0.4;
  //
  // float T = 3;
  // float w0 = PI2/T;
  // float wavLen = length(waves[0].vec);
  //
  // waves[0].freq = float(int(sqrt(9.8*wavLen*hypTan(wavLen*0.35))))*w0;
  // waves[0].phase = 0.123;
  //
  // setWaveVec(waves[1], 4, vec2(45,0));
  // wavLen = length(waves[1].vec);
  // waves[1].amp = 0.6;
  // waves[1].freq = float(int(sqrt(9.8*wavLen*hypTan(wavLen*0.025))))*w0;
  // waves[1].phase = 3.0;
  //
  // setWaveVec(waves[2], 14, vec2(-1,-1));
  // wavLen = length(waves[2].vec);
  // waves[2].amp = 0.4;
  // waves[2].freq = float(int(sqrt(9.8*wavLen*hypTan(wavLen*0.15))))*w0;
  // waves[2].phase = 9.0;
  //
  // return gerstner(vec, time, waves) + vec3(10 + 10*sin(0),-5,10 + 10*cos(0));// + vec3(0,x.y*150.,0);
  //float val = texture2D(tex, (vec2(x.x,x.z)/(5/6.28)+vec2(16))/64).x;

  // newPoint((i)*scale, (j)*scale);
  // newPoint((i)*scale, (j+1.0f)*scale);
  // newPoint((i+1.0f)*scale, (j)*scale);
  // newPoint((i+1.0f)*scale, (j+1.0f)*scale);
  // newPoint((i)*scale, (j+1.0f)*scale);
  // newPoint((i+1.0f)*scale, (j)*scale);

  // heightMapCpu_[(2*i)*x_ + 2*j] = dft((i - x_/2)*2*M_PI/gridlen_, (j - x_/2)*2*M_PI/gridlen_).real();
  // heightMapCpu_[(2*i+1)*x_ + 2*j] = dft((i + 1.0f - x_/2)*2*M_PI/gridlen_, (j - x_/2)*2*M_PI/gridlen_).real();
  // heightMapCpu_[(2*i)*x_ + 2*j+1] = dft((i - x_/2)*2*M_PI/gridlen_, (j + 1.0f - x_/2)*2*M_PI/gridlen_).real();
  // heightMapCpu_[(2*i+1)*x_ + 2*j+1] = dft((i + 1.0f - x_/2)*2*M_PI/gridlen_, (j + 1.0f - x_/2)*2*M_PI/gridlen_).real();

  // we have newpoint, want to get to index so that we can index into heightMapCpu_
  // scale = 6.28/5
  // i, j in newpoint go from negative 16
  // add 16

  vec3 val = texelFetch(tex, int(inPos.w)).xyz;
  //float val = texture2D(tex, (x.xz/8 + vec2(2))/2).x;
  //return vec3(x.x+75, val.x-15, x.z+75);
  return vec3(x.x+val.y+75, val.x-15, x.z+val.z+75);
  //return vec3(x.x+2, sin(time*5*x.z + x.x*5 + x.z*5)*sin(time*3)*0.2 + -abs(cos(time+x.x)*0.5) + sin(time*0.3 + x.x)*sin(time*1 + x.z)*sin(time*0.3) + sin(x.z*x.z + time)*0.3 + 5, x.z+5);
}


void main()
{
    pos = model*vec4(mutateVec(vec3(inPos)), 1.0);
    literalPos = vec4(inPos.w);
    gl_Position = proj * view * model * pos;
}
