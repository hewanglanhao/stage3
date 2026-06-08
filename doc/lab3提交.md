### 开发服务器
#### 开启
```bash
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.34.113:8080/start" `
  -ContentType "application/json" `
  -Body '{"id":"22307110243","gpu":1}'
```

```bash
$r = Invoke-RestMethod -Method GET `
  -Uri "http://10.176.34.113:8080/submit_status/d6352416b19d1d3b06b3eb1fda73350f"

$r | Format-List *
```

```bash
#第二个
ssh -o IdentitiesOnly=yes root@10.176.34.113 -p 47409 -i "C:\Users\lizihan\.ssh\id_mlclass"
```




### 提交服务器
#### 开启
```bash
# test
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.37.31:8080/start" `
  -ContentType "application/json" `
  -Body '{"id":"22307110243","gpu":0}'

```

```bash
# test
$r = Invoke-RestMethod -Method GET `
  -Uri "http://10.176.37.31:8080/submit_status/198e78aac16a32c9a2b710dea5e83136"

$r | Format-List *

```

```bash
# test
ssh -o IdentitiesOnly=yes root@10.176.37.31 -p 36049 -i "C:\Users\lizihan\.ssh\id_mlclass"
```

### 停止环境
```bash
# test
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.37.31:8080/finish" `
  -ContentType "application/json" `
  -Body '{"id":"22307110243"}'

```

### 提交
```bash
# pwsh
$body = @{
  id  = "22307110243"
  gpu = 1
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://10.176.37.31:8080/submit3" `
  -ContentType "application/json" `
  -Body $body

```

### 查询提交状态

```bash
curl http://10.176.37.31:8080/submit_status/9f91ce5b1284d6625eab7bfb118a2a76
```

可能的状态值包括：

* `running`
* `succeeded`
* `failed`
* `killed`
