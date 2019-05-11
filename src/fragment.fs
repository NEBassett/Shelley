#version 330 core
out vec4 color;

in vec4 pos;

void main()
{
  color = vec4(0,0.85,1.0,1)*vec4((pos.y+35)/25);///length(pos.xz);
}
