#define BOOST_THREAD_PROVIDES_FUTURE

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
#include <vector>
#include <complex>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <random>
#include <math.h>
#include <utility>
#include <algorithm>
#include <boost/hana/tuple.hpp>
#include <boost/hana/zip.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/optional.hpp>
#include <boost/memory_order.hpp>
#include <boost/atomic/atomic.hpp>

const std::string g_infoPath   = "../src/config.xml";
const std::string g_vertexPath = "../src/vertex.vs";
const std::string g_fragmentPath = "../src/fragment.fs";

template<typename T>
struct complexVec2
{
  using type = complexVec2<T>;
  std::complex<T> a, b, c;

  auto operator*(const std::complex<T> &coeff) const
  {
    return type{coeff*a, coeff*b};
  }

  auto operator+(const type &other) const
  {
    return type{other.a+a, other.b+b};
  }
};

template<typename T>
struct complexVec3
{
  using type = complexVec3<T>;
  std::complex<T> a, b, c;

  auto operator*(const std::complex<T> &coeff) const
  {
    return type{coeff*a, coeff*b, coeff*c};
  }

  auto operator+(const type &other) const
  {
    return type{other.a+a, other.b+b, other.c+c};
  }

  auto operator-(const type &other) const
  {
    return type{a-other.a, b-other.b, c-other.c};
  }
};

struct vertex
{
  glm::vec4 position;

  vertex()
  { }

  vertex(float x, float y, float z, float w) : position(x,y,z,w)
  { }
};

struct wave
{
  float time, constant, windForce;
  glm::vec2 dir;

  wave()
  { }

  wave(float time_, float A, float V, glm::vec2 direction) :
    time(time_),
    constant(A),
    windForce(V*V/9.8),
    dir(direction)
  { }

};

struct simInfo
{
  wave wav;
  size_t len, gridlen, numThreads = 2;
  float displacementScale;

  simInfo()
  { }

  simInfo(size_t gridlen_, size_t len_, wave wav_, float dispScale = 1.0f) : wav(wav_), len(len_), gridlen(gridlen_), displacementScale(dispScale)
  { }

  simInfo(const std::string &filename)
  {
    boost::property_tree::ptree settings;
    boost::property_tree::xml_parser::read_xml(filename, settings);
    wav = wave(
      settings.get<float>("simInfo.wave.time"),
      settings.get<float>("simInfo.wave.phillipsConstant"),
      settings.get<float>("simInfo.wave.windSpeed"),
      glm::vec2(settings.get<float>("simInfo.wave.windDirectionX"),
      settings.get<float>("simInfo.wave.windDirectionY"))
    );

    displacementScale = settings.get<float>("simInfo.displacementScale");
    gridlen = settings.get<unsigned int>("simInfo.gridLength");
    len = settings.get<unsigned int>("simInfo.simulationLength");
  }
};

class amplitudeGenerator
{
  std::mt19937 gen_;
  std::normal_distribution<> dist_;
  wave wav_;
public:
  using complex = std::complex<float>;
  using vec2 = complexVec2<float>;

  struct fftVertex
  {
    vec2 displacementCoefficient;
    complex h0;
    complex h0Conj;
    float dispersion;

    template<typename T>
    fftVertex(const wave &wav, T&& waveF, float rand0, float rand1, float x, float y)
    {
      auto k = std::max(sqrtf(x*x+y*y), 0.00001f);
      h0 = (1/sqrtf(2)) * (complex(rand0, rand1)) * sqrtf(waveF(x,y));
      h0Conj = std::conj((1/sqrtf(2)) * (complex(rand0, rand1)) * sqrtf(waveF(-x,-y)));
      dispersion = int(sqrt(9.8*k) / ((2*M_PI)/wav.time)) * ((2*M_PI)/wav.time);
      displacementCoefficient = vec2{complex(x*(1/k)),complex(y*(1/k))}*complex(0,-1.0f);
      //std::cout << " factor1squared " << wavXY << '\n';
    }
  };

  amplitudeGenerator(wave wav) : gen_(std::random_device()()), dist_(0,1), wav_(wav)
  { }

  auto newAmp(float x, float y)
  {
    const auto phillips = [this](float x, float y)
    {
      auto k = sqrtf(x*x + y*y);
      if (k < 0.000001) return 0.0f;
      //std::cout << "( k " << k;
      auto cosineFactorSqrt = wav_.dir[0]*x/k + wav_.dir[1]*y/k;
      //std::cout << " cosineFactorSqrt " << k << " )";
      return wav_.constant * (exp(-1/(k*wav_.windForce*k*wav_.windForce))/(k*k*k*k)) * cosineFactorSqrt * cosineFactorSqrt;
    };

    return fftVertex(wav_, phillips, dist_(gen_), dist_(gen_), x, y);
  }
};

struct plane
{
  GLuint buf;
  GLuint obj;
  size_t count;
  std::vector<amplitudeGenerator::fftVertex> amplitudeCoefficients;

  ~plane()
  {
    // glDeleteBuffers(1, &buf);
    // glDeleteVertexArrays(1, &obj);
  }
};

void GLAPIENTRY msgCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam )
{
  //std::cout << message << '\n';
}

auto constructPlane(const simInfo &info)
{
  float scaleK = (2*M_PI)/info.len;
  float scalePoints = info.len/info.gridlen;
  amplitudeGenerator ampGen(info.wav);
  std::vector<vertex> verts;
  std::vector<amplitudeGenerator::fftVertex> amplitudeCoefficients;

  const auto newPoint = [&verts, &amplitudeCoefficients, &ampGen, &info](auto x, auto y, auto index)
  {
    verts.emplace_back(x, 0.0f, y, index);
  };

  for(auto i = -float{float(info.gridlen)/2}; i < float{info.gridlen/2}; i++)
  {
    for(auto j = -float{info.gridlen/2}; j < float{info.gridlen/2}; j++)
    {
      auto ind = amplitudeCoefficients.size();
      if((info.gridlen/2 - j > 0.001) && (info.gridlen/2 - j > 0.001))
      {
        newPoint((i)*scalePoints, (j)*scalePoints, ind); // base
        newPoint((i)*scalePoints, (j+1.0f)*scalePoints, ind+1); // one column over
        newPoint((i+1.0f)*scalePoints, (j)*scalePoints, ind+info.gridlen); // one row over
        newPoint((i+1.0f)*scalePoints, (j+1.0f)*scalePoints, ind+1+info.gridlen); // diagonal
        newPoint((i)*scalePoints, (j+1.0f)*scalePoints, ind+1); // as above
        newPoint((i+1.0f)*scalePoints, (j)*scalePoints, ind+info.gridlen); // as below
      }

      amplitudeCoefficients.emplace_back(ampGen.newAmp(i*scaleK,j*scaleK)); // gridlen*gridlen amplitudes
    }
  }

  GLuint buf, obj;
  glGenBuffers(1, &buf);
  glGenVertexArrays(1, &obj);

  std::cout << "size " << sizeof(vertex) << "\n";

  glBindBuffer(GL_ARRAY_BUFFER, buf);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertex)*verts.size(), verts.data(), GL_STATIC_DRAW);

  glBindVertexArray(obj);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)0);
  glEnableVertexAttribArray(0);

  return plane{buf, obj, info.gridlen*info.gridlen*6, amplitudeCoefficients};
}

template<typename T, typename U, typename V>
void ditfft(const T src, U dest, V &&mutator, size_t N, int s, const size_t srcStride = 1, const size_t destStride = 1)
{
  using valtype = typename std::complex<float>;
  auto halfN = N/2;
  if (halfN == 0)
  {
    *dest = mutator(*src);
    //std::cout << "dest: " << *dest << '\n';
  } else {
    ditfft(src, dest, mutator, halfN, 2*s, srcStride, destStride);
    ditfft(src+s*srcStride, dest+halfN*destStride, mutator, halfN, 2*s, srcStride, destStride);
    for(size_t i = 0; i < halfN; i++)
    {
      auto kInd = (dest+i*destStride);
      auto coeffInd = (dest+(i+halfN)*destStride);
      auto copy = *kInd;
      auto expRes = std::exp(valtype(0.0f, -2*M_PI*i/N));
      //auto expRes = valtype(cos(-6.283185*i/N), sin(-6.283185*i/N)); // e^ix

      *kInd = copy + (*coeffInd)*expRes;
      *coeffInd = copy - (*coeffInd)*expRes;
      //std::cout << "copy: " << copy << " expres: " << expRes << '\n';
    }
  }
}

class fftContext
{
  struct taskControlBlock
  {
    boost::atomic<int> isNew;
  };

  struct task
  {
    const simInfo *info;
    plane *target;
    std::vector<complexVec3<float>> *output;
    double time;
  };

  std::vector<boost::thread> threads_;
  std::vector<complexVec3<float>> horizontalFfts_;
  std::vector<taskControlBlock> taskControlBlocks_;
  boost::optional<task> task_;
  boost::atomic<int> firstStageCount_;
  boost::atomic<int> secondStageCount_;
  boost::mutex mtx_;
  size_t numThreads_;
public:
  fftContext(const fftContext &) = delete;

  fftContext()
  { }

  fftContext(const simInfo &info) :
    threads_(),
    horizontalFfts_(info.gridlen*info.gridlen),
    taskControlBlocks_(info.numThreads),
    task_(boost::none),
    firstStageCount_(),
    secondStageCount_(),
    mtx_(),
    numThreads_(info.numThreads)
  {
    for(auto j = size_t{0}; j < info.numThreads; j++)
    {
      threads_.emplace_back([this, j]()
      {
        while(true)
        {
          auto val = taskControlBlocks_[j].isNew.exchange(0);
          if(val == 0)
          {
            continue;
          }

          if(!task_) // task is null?
          {
            continue; // skip
          }

          auto &task = task_.get();
          auto info = *(task.info);

          for(auto i = size_t{0}; i < info.gridlen/numThreads_; i++) // generate inner ffts over x's
          {
            auto mult = std::complex<float>(pow(-1.0f, i*numThreads_+j), 0);
            ditfft(
              task.target->amplitudeCoefficients.begin() + (i*numThreads_+j)*info.gridlen,
              horizontalFfts_.begin() + (i*numThreads_+j)*info.gridlen,
              [this, info, &mult, &task](const auto &vertex)
                {
                auto coefficient =
                  mult*(std::exp(std::complex<float>(0, vertex.dispersion*task.time))*vertex.h0 +
                  std::exp(std::complex<float>(0, -vertex.dispersion*task.time))*vertex.h0Conj);
                auto vectorComponent = vertex.displacementCoefficient*coefficient*info.displacementScale;
                return complexVec3<float>
                {
                  vectorComponent.a,
                  vectorComponent.b,
                  coefficient
                };
            }, info.gridlen, 1);
          }

          firstStageCount_.fetch_add(1); // increment our horizontal fft counter

          while(firstStageCount_.load() < numThreads_) // all of our horizontal ffts have been completed
          { }

          for(auto i = size_t{0}; i < info.gridlen/numThreads_; i++) // update output
          {
            auto mult = std::complex<float>(pow(-1.0f, i*numThreads_ + j), 0);
            // fft varies over z
            ditfft(horizontalFfts_.begin() + i*numThreads_ + j, task.output->begin() + i*numThreads_ + j, [&mult](const auto &item){
              return item*mult;
            }, info.gridlen, 1, info.gridlen, info.gridlen);
          }

          secondStageCount_.fetch_add(1); // finished
        }
      });

      threads_[j].detach();
    }
  }

  auto fft(const simInfo &info, plane &target, std::vector<complexVec3<float>> &output, double time)
  {
    if(horizontalFfts_.size() < info.gridlen*info.gridlen)
    {
      static constexpr auto zero = std::complex<float>(0,0);
      horizontalFfts_.resize(info.gridlen*info.gridlen, complexVec3<float>{zero,zero,zero});
    }

    boost::lock_guard<boost::mutex> lock(mtx_); // wait until this fft is done before running another
    task_ = task{&info, &target, &output, time};

    firstStageCount_.store(0);
    secondStageCount_.store(0);

    for(auto &block : taskControlBlocks_)
    {
      block.isNew.store(1);
    }

    while(secondStageCount_.load() < numThreads_) // halt on calling thread until the fft is complete
    {}
  }
};



class fftwaterProgram
{
  simInfo simInfo_;
  GLuint program_ = 0;
  GLuint heightmap_ = 0;
  GLuint heightmaptbo_ = 0;
  std::vector<complexVec3<float>> complexMapCpu_;
  std::vector<glm::vec3> heightMapCpu_;
  GLint timeLoc_;
  GLint viewLoc_;
  GLint projLoc_;
  GLint modelLoc_;
  fftContext *fft_;

  template<typename... Ts>
 auto compile(const Ts&... paths)
 {
   program_ = glCreateProgram();

   const auto enums = std::array<unsigned int, 5>{GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER};
   const auto emptyStr = std::string("");
   auto shaders = std::array<GLuint, 5>{0,0,0,0,0};

   std::ifstream input;
   input.exceptions(std::ifstream::failbit|std::ifstream::badbit);

   auto ind = size_t{0};
   boost::hana::for_each(std::tie(paths...), [this, &input, &enums, &shaders, &ind, &emptyStr](const auto& path)
   {
       std::string src;
       auto type = enums[ind];
       decltype(auto) shader = shaders[ind++];

       try
       {
         if(!(path==emptyStr))
         {
           input.open(path);
           std::stringstream stream;
           stream << input.rdbuf();
           src = stream.str();
           input.close();

           const char* srcCStr = src.c_str();
           shader = glCreateShader(type);
           glShaderSource(shader, 1, &srcCStr, NULL);
           glCompileShader(shader);

           GLchar log[1024];
           GLint success;

           glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
           if(!success)
           {
             glGetShaderInfoLog(shader, 1024, NULL, log);
             std::cout << "shader compilation error: " << log << '\n';
           }
           glAttachShader(program_, shader);
         } else {
           return;
         }
       }
       catch(std::ifstream::failure err)
       {
         std::cout << "error reading shader source(s)\n";
       }
   });

   glLinkProgram(program_);
   glUseProgram(program_);

   timeLoc_ = glGetUniformLocation(program_, "time");
   viewLoc_ = glGetUniformLocation(program_, "view");
   modelLoc_ = glGetUniformLocation(program_, "model");
   projLoc_ = glGetUniformLocation(program_, "proj");

   for(auto& shader : shaders)
   {
     glDeleteShader(shader);
   }
 }

public:
  fftwaterProgram()
  {
  }

  template<typename... Ts>
  auto recompile(const Ts&... paths)
  {
    glDeleteProgram(program_);
    compile(paths...);
  }

  auto draw(const glm::mat4 &proj, const glm::mat4 &view, const glm::mat4 &model, double time, plane &toDraw)
  {
    fft_->fft(simInfo_, toDraw, complexMapCpu_, time);

    auto signs = std::array<float,2>{-1.0, 1.0f};

    for(auto i = size_t{0}; i < complexMapCpu_.size(); i++)
    {
      //std::cout << "x: " << i/simInfo_.gridlen << " value: " << complexMapCpu_[i].real() << "\n";
      heightMapCpu_[i] = glm::vec3(
        complexMapCpu_[i].c.real()*signs[(i/simInfo_.gridlen)%2==0],
        complexMapCpu_[i].a.real()*signs[(i/simInfo_.gridlen)%2==0],
        complexMapCpu_[i].b.real()*signs[(i/simInfo_.gridlen)%2==0]
      );
    }

    // heightMapCpu_[x_*x_*4-1] = 50.0f;
    // heightMapCpu_[x_*x_*4-1] = 50.0f;

    glBindBuffer(GL_TEXTURE_BUFFER, heightmaptbo_);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(decltype(heightMapCpu_)::value_type)*heightMapCpu_.size(), heightMapCpu_.data(), GL_STREAM_DRAW);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, heightmap_);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, heightmaptbo_);

    glUseProgram(program_);
    glUniform1f(timeLoc_, time);
    glUniformMatrix4fv(viewLoc_, 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(projLoc_, 1, GL_FALSE,  &proj[0][0]);
    glUniformMatrix4fv(modelLoc_, 1, GL_FALSE,  &model[0][0]);

    glBindVertexArray(toDraw.obj);
    glDrawArrays(GL_TRIANGLES, 0, toDraw.count);
  }

  template<typename... Ts>
  fftwaterProgram(const simInfo &info, fftContext& fft, const Ts&... paths) :
   simInfo_(info),
   program_(0),
   heightmap_(0),
   heightmaptbo_(0),
   complexMapCpu_(info.gridlen*info.gridlen),
   heightMapCpu_(info.gridlen*info.gridlen),
   timeLoc_(0),
   viewLoc_(0),
   projLoc_(0),
   modelLoc_(0),
   fft_(&fft)
  {
    glGenBuffers(1, &heightmaptbo_);
    glBindBuffer(GL_TEXTURE_BUFFER, heightmaptbo_);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(float)*heightMapCpu_.size(), nullptr, GL_STREAM_DRAW);

    glGenTextures(1, &heightmap_);
    glBindTexture(GL_TEXTURE_BUFFER, heightmap_);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, heightmaptbo_);

    compile(paths...);
  }

  fftwaterProgram(fftwaterProgram &&other) :
    simInfo_(other.simInfo_),
    program_(other.program_),
    heightmap_(other.heightmap_),
    heightmaptbo_(other.heightmaptbo_),
    heightMapCpu_(std::move(other.heightMapCpu_)),
    timeLoc_(other.timeLoc_),
    viewLoc_(other.viewLoc_),
    projLoc_(other.projLoc_),
    modelLoc_(other.modelLoc_),
    fft_(other.fft_)
  {
    other.program_ = 0;
    other.heightmap_ = 0;
    other.heightmaptbo_ = 0;
  }

  auto operator=(fftwaterProgram &&other)
  {
    if(this != &other)
    {
      glDeleteBuffers(1, &heightmaptbo_);
      glDeleteTextures(1, &heightmap_);
      glDeleteProgram(program_);

      heightmaptbo_ = 0;
      heightmap_ = 0;
      program_ = 0;

      std::swap(simInfo_, other.simInfo_);
      std::swap(program_, other.program_);
      std::swap(heightmap_, other.heightmap_);
      std::swap(heightmaptbo_, other.heightmaptbo_);
      std::swap(complexMapCpu_, other.complexMapCpu_);
      std::swap(heightMapCpu_, other.heightMapCpu_);
      std::swap(timeLoc_, other.timeLoc_);
      std::swap(viewLoc_, other.viewLoc_);
      std::swap(projLoc_, other.projLoc_);
      std::swap(modelLoc_, other.modelLoc_);
      std::swap(fft_, other.fft_);
    }
  }

  ~fftwaterProgram()
  {
    glDeleteBuffers(1, &heightmaptbo_);
    glDeleteTextures(1, &heightmap_);
    glDeleteProgram(program_);

    heightmaptbo_ = 0;
    heightmap_ = 0;
    program_ = 0;
  }
};



fftContext fft{simInfo(g_infoPath)};
fftwaterProgram g_prog;


int main()
{
  glfwSetErrorCallback([](auto err, const auto* desc){ std::cout << "Error: " << desc << '\n'; });

  // glfw init
  if(!glfwInit())
  {
    std::cout << "glfw failed to initialize\n";
    std::exit(1);
  }

  // context init
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  auto window = glfwCreateWindow(640, 480, "FFT Water", NULL, NULL);
  if (!window)
  {
    std::cout << "window/glcontext failed to initialize\n";
    std::exit(1);
  }

  glfwMakeContextCurrent(window);

  // glew init
  auto err = glewInit();
  if(GLEW_OK != err)
  {
    std::cout << "glew failed to init: " << glewGetErrorString(err) << '\n';
    std::exit(1);
  }

  // gl init
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(msgCallback, 0);
  glEnable(GL_DEPTH_TEST);

  // fft program setup
  auto sim = simInfo{g_infoPath};
  g_prog = fftwaterProgram(sim, fft, g_vertexPath, g_fragmentPath);

  glfwSetKeyCallback(window, [](auto window, auto key, auto scancode, auto action, auto mods){
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    if(key == GLFW_KEY_F4)
    {
      g_prog = fftwaterProgram(simInfo{g_infoPath}, fft, g_vertexPath, g_fragmentPath);
      std::cout << "f4 received\n";
    }

  });

  // pre loop declarations/actions
  int width, height;
  glm::mat4 view, proj;
  auto plane = constructPlane(sim);

  view = glm::lookAt(glm::vec3(0,7,0), glm::vec3(10,0,10), glm::vec3(0,1,0));

  glfwSwapInterval(1);
  while(!glfwWindowShouldClose(window))
  {
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    proj = glm::perspective(1.57f, float(width)/float(height), 0.1f, 7000.0f);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    g_prog.draw(proj, view, glm::mat4(), glfwGetTime(), plane);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  std::exit(0);
  return 0;
}
