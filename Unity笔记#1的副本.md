# 游戏生命周期调用顺序：#
awake在物体gameObject激活时就调用,仅调用一次；
OnEnable在组件即脚本激活后调用，可重复调用；
Start在Update之前，OnEnable之后调用，也是唯一调用；
Update每帧调用一次，按不同时间间隔调用；
FixedUpdate是按相同时间间隔调用；
LateUpdate紧接着Update调用；
OnDisable在取消组件激活时调用，可重复调用；
OnDestroy在组件销毁即移除脚本时调用；
若一个游戏物体里有多个脚本，统一将各自脚本的awake执行完后再执行OnEnable，其余顺序依次类推，不同脚本的awake等生命周期的执行顺序在没有设定脚本执行顺序是随机的。

![img](https://img-blog.csdnimg.cn/20190916105139500.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE3MzQ3MzEz,size_16,color_FFFFFF,t_70)

vector3可以代表向量，坐标，旋转，和缩放等。
创建语句如：Vector3 v = new Vector3(1,1,1);
v = Vector3.zero;//(0,0,0)
v = Vector3.one;//(1,1,1)
Vector3 v = Vector3.forward;
计算向量夹角：Vector3.Angle(v,v2);
计算两点之间距离：Vector3.Distance(v,v2);
向量点乘：Vector3.Dot(v,v2);
向量叉乘：Vector3.Cross(v,v2);
向量插值：Vector3.Lerp(Vector3.zero,Vector3.one,0.5f);//两者相加后乘第三位
向量模长：v.magnitude;
向量单位化：v.normalized;

方向有两种表示方式：欧拉角和四元数。
unity中根据自身坐标系旋转的顺序是y-x-z,根据世界坐标系旋转的顺序是z-x-y。

y轴向上，x轴向右，z轴向前。

Vector3 v = new Vector3(0,30,0);
Quaternion q = Quaternion identity;//初始化q为如欧拉角(0,0,0)的向量
q = Quaternion.Euler(v);//把欧拉角转换为四元数进行初始化
v = q.eulerAngles;//把四元数转化为欧拉角
q = Quaternion.LookRotation(new Vector3(0,0,0));//看向一个位置，向该位置进行旋转

Debug.Log();//普通信息
Debug.LogWarning();//警告信息
Debug.LogError();//报错信息
Debug.DrawLine(Vector3.zero,Vector3.one,Color.red);//绘制一条从点(0,0,0)到点(1,1,1)的红色线段，如果没加颜色默认为白色
Debug.DrawRay(new Vector3(1,0,0),new Vector3(1,1,0),Color.red);//绘制一条从点(1,0,0)到以点(1,0,0)为基点坐标位置为(1,1,0)的点的红色线段

gameObject指代当前脚本所挂载的游戏物体
gameObject.name：游戏物体名称
gameObject.tag：游戏物体标签
gameObject.layer：游戏物体图层
gameObject.activeInHierachy：游戏物体真正的激活状态，灰色就是没激活
gameObject.activeSelf：游戏物体自身激活状态，有对勾就是激活了

transform组件可直接调用，同gameObject，如：
Debug.Log(transform.position);

在当前物体获取其他组件，如：
BoxCollider bc = GetComponent<BoxCollider>();
获取当前物体子物体身上的组件，如： 
GetComponentInChildren<CapsuleCollider>(bc);
获取当前物体父物体身上的组件，如：
GetComponentInParent<CapsuleCollider>(bc);
在当前物体添加其他组件，如：
gameObject.AddComponent<AudioSource>();

获取游戏物体的多种方式：
public GameObject Cube;
然后将Cube拖动到游戏对象中。

GameObject test = GameObject.Find("Test");//通过游戏物体名称获取游戏物体
test = GameObject.FindWithTag("enemy");//通过游戏物体标签获取游戏物体

test.SetActive(false);//取消游戏物体激活状态

public GameObject Prefab;
Instantiate(Prefab);//通过预设体来实例化一个游戏物体
Instantiate(Prefab,transform);//通过预设体来实例化一个当前脚本所挂载的物体下的子物体
GameObject go = Instantiate(Prefab,Vector3.zero,Quaternion.identity);//将新生成的游戏物体置于(0,0,0)的位置且无旋转
Destroy(go);//销毁游戏物体

Time.time 游戏开始到现在所花时间
Time.timeScale 游戏时间缩放值（默认为1.0）
Time.fixedDeltaTime 游戏时间固定间隔（默认为0.02秒）
Time.deltaTime 上一帧到这一帧所用游戏时间

Application.dataPath 游戏数据文件夹路径（只读且加密压缩）
Application.persistentDataPath 游戏数据持久化文件夹路径（可写文件，不同平台通用）
Application.streamingAssetsPath     StreamingAssets文件夹路径（ 只读但不会加密压缩，比如配置文件，默认为Assets文件夹下的StreamingAssets文件夹）
Application.temporaryCachePath  临时文件夹路径（ 默认为系统分配的临时文件夹）
Application.runInBackground 控制是否在后台运行
Application.OpenURL(" ");//打开网址
Application.Quit();//退出游戏 

SceneManager.LoadScene();//场景跳转，可以通过编号和名称进行索引，需要引用UnityEngine.SceneManagement头文件
Scene scene = SceneManager.GetActiveScene();//获取当前激活场景
scene.name 场景名称
scene.isLoaded 判断场景是否已经加载
scene.path 场景路径
scene.buildIndex 场景索引序号
scene.GetRootGameObjects();//生成一个GameObject数组，获取当前场景内所有游戏物体
SceneManager.sceneCount 当前已加载场景数量
SceneManager.CreateScene("newScene");//创建新场景
SceneManager.UnloadSceneAsync(newScene);//卸载场景
SceneManager.LoadScene("MyScene",LoadSceneMode.Single);//同上场景跳转
SceneManager.LoadScene("MyScene",LoadSceneMode.Additive);//场景叠加 

# 异步加载场景：#
![请添加图片描述](https://img-blog.csdnimg.cn/3c0589c04a0040c2832e99850c84edfa.png)![在这里插入图片描述](https://img-blog.csdnimg.cn/772d83d420504be1b0907f3469e1233e.png#pic_center)

transform.postion:世界位置

transform.localPosition:相对位置

transform.rotation:物体旋转四元数

transform.localPosition:相对父物体旋转四元数

transform.eulerAngles:物体旋转欧拉角

transform.localEulerAngles:相对父物体旋转欧拉角

transform.localScale:相对父物体缩放

transform.forward:获取向前向量

transform.up:获取向上向量

transform.right:获取向右向量

transform.LookAt();//固定看向某一点

transform.Rotate(Vector3.up,1);//每一帧绕y轴旋转一度

transform.RotateAround(Vector3.zero,Vector3.up,1);//每一帧绕原点的y轴旋转一度

transform.Translate(Vector3.forward * 0.1f);//每一帧向前移动

transform.parent.gameObject:获取父物体

transform.childCount:获取子物体个数

transform.DetachChildren();//解除与所有子物体的父子关系

Transform trans = transform.Find("Child");//获取名称为Child的子物体，注意返回值为Transform类型

transform.GetChild(0);//获取第0个子物体，返回值为Transform类型

bool res = trans.IsChildOf(transform);//判断一个物体是不是另一个物体的子物体，返回值为布尔型

trans.SetParent(transform);//把tranifsform设置为trans的父物体



