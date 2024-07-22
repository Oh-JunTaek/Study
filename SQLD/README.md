# SQLD 자격증 

## 이론 강의 https://www.youtube.com/watch?v=lxiEiAjp7d0&list=PL6i7rGeEmTvpLoDkB-kECcuD1zDt_gaPn

## 1강 데이터베이스
* 데이터베이스 : 데이터들을 모아 통합적으로 관리하는 기술
ex) 로그인의 구현
* SQL[Structured Query Language] : 관계형 데이터베이스 관리 시스템(RDBMS)에서 데이터를 정의하고 조작하기 위한 표준 프로그래밍 언어

* 요구사항 접수 -> 개념적 데이터 모델링 -> 논리적 데이터 모델링 -> 물리적 데이터 모델링 ->데이터베이스에 저장할 수 있게 세팅
* 엔터티[ENTITY] : 업무에 필요한 정보를 저장/관리하기 위한 집합적인 명사 개념적
* 인스턴스 [INSTANCE] : 엔터티 집합 내에 존재하는 개별적인 대상

### 엔터티
 ** 엔터티의 특징 **
  - 반드시 업무에서 필요한 대상이고 업무에 사용될 것.
  - 유일한 식별자로 식별이 가능할 것
  - 인스턴스가 2개 이상일 것
  - 속성이 반드시 2개 이상 존재할 것
  - 관계가 하나 이상 존재할 것

 * 엔터티의 분류(유무형에 따라 분류)
  - 유형[Tangible](물리적 형태가 있는 엔터티) ex_직원 주류 강사
  - 개념[Conceptual](물리적 형태가 없는 엔터티) ex_부서 과먹 계급
  - 사건[Event](업무 수행 중 발생하는 엔터티) ex_강의 매출 주문 상담\

 ** 엔터티의 분류(발생시점에 따라 분류) **
  - 기본/키
  - 중심
  - 행위

 ** 명명규칙 **
  - 현업용어 사용 권장 (사람->고객)
  - 약어 X (일매목 ->일별매출정보)
  - 단수명사(직원들->직원)
  - 유일한 엔터티이름
  - 생성의미대로 이름을 부여(연락처목록 -> 직원/고객 연락처 목록)

### 속성[ATTRIBUTE]
업무상 관리하기 위해 의미적으로 더는 부닐되지 않는 최소의 데이터 단위

* 식별자 : 엔터티 내 유일한 인스턴스를 식별할 수 있는 속성의 집합
ex_ 직원ID 겹치지 않으면 OK/이름은 동명이인이 있으면 X

** 속성의 분류(특성에 따른 분류) **
 - 기본 : 업무로부터 추출한 속성으로 제일 많이 발생
 - 설계[Designed] : 규칙화등이 필요해 만든 속성ex_코드성,일련번호
 - 파생[Derived] : 다른 속성들로부터 계산/변형되어 만들어진 속성. 정보가 추가될 때 마다 변경이 잦음.
 ```sql
 CREATE TABLE Student (
    StudentID INT PRIMARY KEY, -- 기본키 속성
    Name VARCHAR(100), -- 일반 속성
    DateOfBirth DATE, -- 일반 속성
    Address VARCHAR(255), -- 일반 속성
    CreatedAt TIMESTAMP, -- 설계 속성
    UpdatedAt TIMESTAMP -- 설계 속성
 );
 ```

** 구성 방식에 따른 분류 **
 - PK [PrimareyKey] : 고유하게 식별할 수 있는 속성. 비어있을 수 없음
 - FK [ForeignKey] : 다른 엔터티의 기본키를 참조하는 속성. 외래키를 통해 엔터티간의 관계를 정의
 - 일반[Regular] : 기본이나 외래키가 아닌 일반속성 ex_이름 주소 전번
 - 복합[Composite] : 두 개 이상의 속성이 결합되어 하나의 속성을 이루는 경우 ex_주소는 도시 도로명 우편번호 등으로 나뉘어짐
 ```sql
 CREATE TABLE Enrollment (
    EnrollmentID INT PRIMARY KEY, -- 기본키 속성
    StudentID INT, -- 외래키 속성
    CourseID INT, -- 외래키 속성
    EnrollmentDate DATE, -- 일반 속성
    Grade CHAR(1), -- 일반 속성
    FOREIGN KEY (StudentID) REFERENCES Student(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Course(CourseID)
 );
 ```