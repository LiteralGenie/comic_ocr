from typing import cast

from doctr.datasets import VOCABS

HANGUL_SYLLABLES = "가각간갇갈갉갊감갑값갓강갖갗같갚갛갔갸갹갼걀걈걋걍거걱건걷걸걹걺검겁것겅겆겉겊겋겄겨격견겯결겸겹겻경곁겪겼고곡곤곧골곪곬곯곰곱곳공곶곺교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂규균귤귬귱그극근귿글긁긇금급긋긍기긱긴긷길긺김깁깃깅깆깇깉깊개객갠갤갬갭갯갱갰걔걘걜게겍겐겔겜겝겟겡겠계곈곌곕곗괴괵괸괼굄굅굇굉굈귀귁귄귈귐귑귓긔과곽관괃괄괆괌괍괏광괐궈궉권궐궘궝궜괘괙괜괠괩괭괬궤궥궷나낙낛난낟날낡낢남납낫낭낮낯낱낳낚났냐냑냔냘냠냡냥너넉넋넌널넒넓넘넙넛넝넢넣넊넜녀녁년녈념녑녓녕녘녔노녹논놀놂놈놉놋농높놓뇨뇩뇬뇰뇸뇹뇻뇽누눅눈눋눌눔눕눗눙눞뉴뉵뉸뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪니닉닌닐닒님닙닛닝닢내낵낸낼냄냅냇냉냈냬네넥넨넬넴넵넷넹넸녜녠뇌뇐뇔뇜뇝뇟뉘뉜뉠뉨뉩뉭늬늰늴늼닁놔놘놜놧놨눠눨눳눴놰눼다닥단닫달닭닮닯닲닳담답닷당닺닻닾닿닦닸댜더덕던덛덜덞덟덤덥덧덩덫덮덯덖덨뎌뎐뎔뎡뎠도독돈돋돌돎돐돔돕돗동돛돝됴두둑둔둘둠둡둣둥듀듄듈듐듕드득든듣들듥듦듧듬듭듯등디딕딘딛딜딤딥딧딩딪딮딨대댁댄댈댐댑댓댕댔댸데덱덴덷델뎀뎁뎃뎅뎄뎨뎬되된될됨됩됫됭됬뒤뒥뒨뒬뒴뒵뒷뒹듸듼딀딉딍돠돤돨둬둰둴둼둿뒀돼됀됄됐뒈뒝라락란랄람랍랏랑랒랖랗랐랴략랸랼럄럅럇량러럭런럴럼럽럿렁렆렇렀려력련렬렴렵렷령렸로록론롤롬롭롯롱롶료룐룔룜룝룟룡루룩룬룰룸룹룻룽류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링맆래랙랜랠램랩랫랭랬럐레렉렌렐렘렙렛렝렜례롄롈롑롓뢰뢴뢸룀룁룃룅룄뤼뤽륀륄륌륏륑릐릔릘릠롸롼뢉뢍뤄뤘뢔뢨뤠마막만많맏말맑맒맘맙맛망맞맟맡맣먀먁먄먈먐먕머먹먼멀멁멂멈멉멋멍멎멓멌며멱면멸몀몁몃명몇몄모목몫몬몯몰몲몸몹못몽뫃묘묜묠묩묫무묵문묻물묽묾뭄뭅뭇뭉뭍뭏묶뮤뮥뮨뮬뮴뮷뮹므믄믈믐믑믓믕미믹민믿밀밂밈밉밋밍및밑밌매맥맨맬맴맵맷맹맺맸먜메멕멘멜멤멥멧멩멨몌몐뫼묀묄묌묍묏묑뮈뮌뮐믜믠믬뫄뫈뫙뫘뭐뭔뭘뭠뭡뭣뭤뫠뭬바박밗반받발밝밞밟밤밥밧방밭밖뱌뱍뱐뱜뱝버벅번벋벌벍벎범법벗벙벚벜벘벼벽변별볌볍볏병볓볕볐보복본볼봄봅봇봉봏볶뵤뵨뵬부북분붇불붉붊붐붑붓붕붙붚뷰뷴뷸븀븁븃븅브븍븐블븜븝븟븡비빅빈빌빎빔빕빗빙빚빛배백밴밷밸뱀뱁뱃뱅뱉뱄뱨베벡벤벧벨벰벱벳벵벴볘볜뵈뵉뵌뵐뵘뵙뵜뷔뷕뷘뷜뷩븨븬븰븽봐봔봡봣봤붜붤붯붴붰봬봰뵀붸사삭삯산삳살삵삶삼삽삿상샅샀샤샥샨샬샴샵샷샹서석섟선섣설섦섧섬섭섯성섶섞섰셔셕션셜셤셥셧셩셨소속손솓솔솖솜솝솟송솥솎쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲슈슉슌슐슘슙슛슝스슥슨슬슭슲슳슴습슷승시식신싣실싫심십싯싱싶새색샌샐샘샙샛생샜섀섄섈섐섕세섹센셀셈셉셋셍셑셒셌셰셴셸솅쇠쇡쇤쇨쇰쇱쇳쇵쇴쉬쉭쉰쉴쉼쉽쉿슁싀싄솨솩솬솰솻솽숴쉈쇄쇈쇌쇔쇗쇘쉐쉑쉔쉘쉠쉡쉥자작잔잖잗잘잚잠잡잣장잦잤쟈쟉쟌쟎쟐쟘쟙쟝저적전절젊점접젓정젖젔져젹젼졀졈졉졋졍졌조족존졸졺좀좁좃종좆좇좋죠죡죤죨죰죵주죽준줄줅줆줌줍줏중쥬쥰쥴쥼즁즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚재잭잰잴잼잽잿쟁쟀쟤쟨쟬제젝젠젤젬젭젯젱젶젰졔졘졜죄죈죌죔죕죗죙죘쥐쥑쥔쥗쥘쥠쥡쥣즤좌좍좐좔좝좟좡줘줬좨좽좼줴줸줼쥄쥅쥈차착찬찮찰참찹찻창찾찼챠챤챦챨챰챱챵처척천철첨첩첫청첬쳐쳑쳔쳘쳤초촉촌촐촘촙촛총쵸쵼춀춈추축춘춛출춤춥춧충츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭채책챈챌챔챕챗챙챘챼체첵첸첼쳄쳅쳇쳉쳈쳬쳰촁최쵠쵤쵬쵭쵯쵱취췬췰췸췹췻췽츼촤촥촨촬촹춰췃췄쵀쵄췌췐카칵칸칼캄캅캇캉캎캈캬캭캰캼캽컁커컥컨컫컬컴컵컷컹컽컾컸켜켠켤켬켭켯켱켰코콕콘콜콤콥콧콩쿄쿠쿡쿤쿨쿰쿱쿳쿵큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹킾캐캑캔캘캠캡캣캥캪캤컈케켁켄켈켐켑켓켕켸쾨쾰퀴퀵퀸퀼큄큅큇큉킈콰콱콴콸쾀쾅쿼퀀퀄퀑쾌쾐쾔쾡퀘퀙퀠퀭타탁탄탈탉탐탑탓탕탚탔탸탼턍터턱턴털턺텀텁텃텅텄텨텬텼토톡톤톨톰톱톳통톺툐투툭툰툴툼툽툿퉁튜튠튤튬튱트특튼튿틀틂틈틉틋틍티틱틴틸팀팁팃팅태택탠탤탬탭탯탱탶탰턔테텍텐텔템텝텟텡텦톄톈퇴퇸툇툉튀튁튄튈튐튑튕틔틘틜틤틥톼퇀퉈퉜퇘퉤퉨퉸파팍판팔팖팜팝팟팡팥팎팠퍄퍅퍼퍽펀펄펌펍펏펑펐펴펵편펼폄폅폇평폈포폭폰폴폼폽폿퐁표푠푤푭푯푸푹푼푿풀풂품풉풋풍퓨퓬퓰퓸퓻퓽프픈플픔픕픗픙피픽핀필핌핍핏핑패팩팬팰팸팹팻팽팼퍠페펙펜펠펨펩펫펭펲폐폔폘폡폣푀푄퓌퓐퓔퓜퓟픠픤퐈퐝풔풩하학한할핥함합핫항햐향허헉헌헐헒헕헗험헙헛헝혀혁현혈혐협혓형혔호혹혼혿홀홅홈홉홋홍홑효횬횰횹횻후훅훈훌훑훔훕훗훙휴휵휸휼흄흇흉흐흑흔흖흗흘흙흝흠흡흣흥흩히힉힌힐힘힙힛힝해핵핸핼햄햅햇행했햬헤헥헨헬헴헵헷헹헸혜혠혤혭회획횐횔횝횟횡휘휙휜휠휨휩휫휭희흰흴흼흽힁화확환활홤홥홧황훠훡훤훨훰훵홰홱홴횃횅횄훼훽휀휄휑까깍깐깓깔깖깜깝깟깡깥깎깠꺄꺅꺈꺌꺼꺽껀껄껌껍껏껑꺾껐껴껸껼꼇꼍꼈꼬꼭꼰꼱꼲꼴꼼꼽꼿꽁꽂꽃꾜꾸꾹꾼꾿꿀꿇꿈꿉꿋꿍꿎뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑깨깩깬깰깸깹깻깽깼꺠께껙껜껠껨껩껫껭껬꼐꾀꾁꾄꾈꾐꾑꾕뀌뀐뀔뀜뀝뀡꽈꽉꽌꽐꽛꽝꽜꿔꿘꿜꿥꿧꿩꿨꽤꽥꽨꽬꽹꿰꿱꿴꿸뀀뀁뀅뀄따딱딴딸딿땀땁땃땅땋딲땄땨땰떠떡떤떨떪떫떰떱떳떵떻떴뗘뗬또똑똔똘똠똡똣똥뚀뚜뚝뚠뚤뚫뚬뚭뚱뜌뜨뜩뜬뜯뜰뜸뜹뜻뜽띠띡띤띨띰띱띳띵때땍땐땔땜땝땟땡땠떼떽뗀뗄뗌뗍뗏뗑뗐뙤뙨뛰뛴뛸뜀뜁뜅띄띅띈띌띔띕띙똬똰똴뚸뛌뙈뙉뛔빠빡빤빨빪빰빱빳빵빻빴뺘뺙뺜뺨뻐뻑뻔뻗뻘뻠뻣뻥뻤뼈뼉뼘뼙뼛뼝뼜뽀뽁뽄뽈뽐뽑뽓뽕뾰뿅뿌뿍뿐뿔뿜뿝뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥빼빽뺀뺄뺌뺍뺏뺑뺐뺴뻬뻭뻰뻴뻼뼁뾔쀠쁴뽜뿨싸싹싻싼쌀쌈쌉쌋쌍쌓쌌쌰쌴쌸썅써썩썬썰썲썸썹썻썽썪썼쎠쏘쏙쏜쏟쏠쏢쏨쏩쏫쏭쑈쑌쑐쑘쑝쑤쑥쑨쑬쑴쑵쑹쓔쓘쓧쓩쓰쓱쓴쓸쓺쓿씀씁씅씨씩씬씯씰씸씹씻씽씼쌔쌕쌘쌜쌤쌥쌧쌩쌨썌쎄쎅쎈쎌쎔쎕쎙쎼쏀쐬쐭쐰쐴쐼쐽쑀쒸쒼씌씐씔씜쏴쏵쏸쏼쐇쐉쐈쒀쒔쐐쐑쐤쒜쒠쒭짜짝짠짢짤짧짬짭짯짱짰쨔쨘쨤쨩쩌쩍쩐쩔쩗쩜쩝쩟쩡쩠쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫒쫓쫗쬬쬰쬼쭁쭈쭉쭌쭐쭘쭙쭛쭝쮸쯀쯔쯕쯘쯜쯤쯧쯩쯪찌찍찐찔찜찝찟찡찢찦찧째짹짼쨀쨈쨉쨋쨍쨌쨰쨴쩨쩩쩬쩰쩸쩹쩽쪠쬐쬔쬘쬠쬡쬤쮜쯰쯴쫘쫙쫜쫠쫭쫬쭤쭹쭸쫴쬈쮀아악안앉않알앍앎앒앓암압앗앙앝앞앟았야약얀얃얄얇얌얍얏양얕얗얐어억언얹얻얼얽얾엄업없엇엉엊엌엎엏었여역연엳열엶엷염엽엾엿영옅옆옇엮였오옥온올옭옮옰옳옴옵옷옹옻옾요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅유육윤율윰윱윳융윷으윽은읃을읅읊음읍읏응읒읓읔읕읖읗이익인일읽읾잃임입잇잉잊잎있애액앤앨앰앱앳앵앴얘얜얠얩에엑엔엘엠엡엣엥엤예옌옐옘옙옛옝옜외왹왼욀욈욉욋욍위윅윈윌윔윕윗윙의읜읠읨읫와왁완왇왈왐왑왓왕왔워웍원월웜웝웟웡웠왜왝왠왬왯왱웨웩웬웰웸웹웻웽윁"
HANGUL_SYLLABLES = cast(list[str], [char for char in HANGUL_SYLLABLES])

KOREAN_ALPHABET = HANGUL_SYLLABLES + [
    char for char in VOCABS["digits"] + VOCABS["ascii_letters"]
]
